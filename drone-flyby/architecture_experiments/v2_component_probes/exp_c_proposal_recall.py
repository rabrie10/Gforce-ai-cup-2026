"""Probe C - class-agnostic proposal recall under a bounded budget.

The question discovery has to answer is "where should the camera look next",
not "which of sixteen classes is this at four pixels". So the metric is recall
against a proposal budget, reported two ways:

``iou50``
    A proposal matches the GT box at IoU >= 0.50. This is what the recognizer
    needs if the proposal box is used directly as the emitted box.
``center``
    A proposal merely contains the GT centre. This is what the *scheduler*
    needs: a region worth pointing the camera at. On a four-pixel object IoU
    0.50 is a box-regression test, not a discovery test, so reporting only
    ``iou50`` would understate a usable discovery path.

Backends are measured at each resolution level, because the level changes both
the object's apparent size and how much of the frame a single view covers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtos import SOURCE_REGION_SIZES  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402
from v2.config import ProposalConfig  # noqa: E402
from v2.geometry import (  # noqa: E402
    LEVEL_DOWNSCALE,
    box_center,
    box_short_side,
    contains,
    iou,
    render_level_view,
    source_to_view,
    source_region,
)
from v2.proposals import ProposalEngine  # noqa: E402


BUDGETS = (10, 20, 40, 80)
SIZE_EDGES = (0.0, 4.0, 8.0, 16.0, 32.0, float('inf'))
SIZE_LABELS = ('<4', '4-<8', '8-<16', '16-<32', '>=32')


def size_bucket(short_side: float) -> str:
    for index in range(len(SIZE_LABELS)):
        if SIZE_EDGES[index] <= short_side < SIZE_EDGES[index + 1]:
            return SIZE_LABELS[index]
    return SIZE_LABELS[-1]


def view_centres(level: int) -> List[tuple]:
    """Camera centres that tile the frame at a level, clamped to legal bounds."""
    width, height = SOURCE_REGION_SIZES[level]
    if level == 0:
        return [(1920, 1080)]
    xs = list(range(width // 2, 3840 - width // 2 + 1, width)) or [1920]
    ys = list(range(height // 2, 2160 - height // 2 + 1, height)) or [1080]
    return [(x, y) for y in ys for x in xs]


def run_backend(
    backend: str,
    scene: str,
    levels: Sequence[int],
    max_budget: int,
) -> dict:
    config = ProposalConfig(backend=backend, budget=max_budget)
    engine = ProposalEngine(config)
    engine.warmup()

    results: Dict[str, dict] = {}
    for level in levels:
        matched = {
            budget: {'iou50': 0, 'center': 0} for budget in BUDGETS
        }
        per_size = {
            budget: defaultdict(lambda: [0, 0, 0]) for budget in BUDGETS
        }
        total = 0
        proposal_counts: List[int] = []
        latencies: List[float] = []

        for frame in frame_numbers(scene):
            image = load_frame(frame, scene)
            annotations = load_annotations(frame, scene)
            for center_x, center_y in view_centres(level):
                region = source_region(level, center_x, center_y)
                view = render_level_view(image, level, center_x, center_y)

                visible = []
                for annotation in annotations:
                    box = tuple(float(value) for value in annotation['bbox'])
                    if not contains(region, box):
                        continue
                    visible.append((annotation['object_id'], box))
                if not visible:
                    continue

                started = time.perf_counter()
                proposals = engine.propose(view, budget=max_budget, level=level)
                latencies.append((time.perf_counter() - started) * 1000.0)
                proposal_counts.append(len(proposals))
                boxes = [proposal.box for proposal in proposals]

                for _, box in visible:
                    total += 1
                    local = source_to_view(box, region)
                    centre = box_center(local)
                    bucket = size_bucket(box_short_side(box) / LEVEL_DOWNSCALE[level])
                    best_iou_rank = None
                    best_center_rank = None
                    for rank, candidate in enumerate(boxes):
                        if best_iou_rank is None and iou(candidate, local) >= 0.50:
                            best_iou_rank = rank
                        if best_center_rank is None and (
                            candidate[0] <= centre[0] <= candidate[2]
                            and candidate[1] <= centre[1] <= candidate[3]
                        ):
                            best_center_rank = rank
                        if best_iou_rank is not None and best_center_rank is not None:
                            break
                    for budget in BUDGETS:
                        per_size[budget][bucket][2] += 1
                        if best_iou_rank is not None and best_iou_rank < budget:
                            matched[budget]['iou50'] += 1
                            per_size[budget][bucket][0] += 1
                        if best_center_rank is not None and best_center_rank < budget:
                            matched[budget]['center'] += 1
                            per_size[budget][bucket][1] += 1

        results[str(level)] = {
            'gt_appearances_in_view': total,
            'mean_proposals': float(np.mean(proposal_counts)) if proposal_counts else 0.0,
            'latency_ms_mean': float(np.mean(latencies)) if latencies else 0.0,
            'latency_ms_p95': float(np.percentile(latencies, 95)) if latencies else 0.0,
            'recall': {
                str(budget): {
                    'iou50': matched[budget]['iou50'] / total if total else 0.0,
                    'center': matched[budget]['center'] / total if total else 0.0,
                }
                for budget in BUDGETS
            },
            'recall_by_size': {
                str(budget): {
                    label: {
                        'iou50': per_size[budget][label][0] / per_size[budget][label][2],
                        'center': per_size[budget][label][1] / per_size[budget][label][2],
                        'n': per_size[budget][label][2],
                    }
                    for label in SIZE_LABELS
                    if per_size[budget].get(label, [0, 0, 0])[2]
                }
                for budget in BUDGETS
            },
        }
        recall = results[str(level)]['recall']
        print(
            f'  L{level}: n={total:4d}  mean proposals {results[str(level)]["mean_proposals"]:5.1f}  '
            f'{results[str(level)]["latency_ms_mean"]:6.1f} ms  '
            + '  '.join(
                f'b{budget}: iou {recall[str(budget)]["iou50"]:.3f} / ctr {recall[str(budget)]["center"]:.3f}'
                for budget in BUDGETS
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--backends', default='saliency,yolo,hybrid')
    parser.add_argument('--levels', default='0,1,2')
    parser.add_argument('--max-budget', type=int, default=max(BUDGETS))
    parser.add_argument('--output', default=str(Path(__file__).parent / 'exp_c_results.json'))
    arguments = parser.parse_args()

    levels = [int(value) for value in arguments.levels.split(',')]
    report = {
        'protocol': {
            'scene': arguments.scene,
            'budgets': list(BUDGETS),
            'levels': levels,
            'note': (
                'Recall is computed only over GT objects fully inside the rendered '
                'view. "center" means a proposal contains the GT centre, which is '
                'the criterion the camera scheduler actually needs.'
            ),
        },
        'backends': {},
    }
    for backend in [value.strip() for value in arguments.backends.split(',') if value.strip()]:
        print(f'\n=== backend: {backend} ===')
        report['backends'][backend] = run_backend(
            backend, arguments.scene, levels, arguments.max_budget
        )

    Path(arguments.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nWrote {arguments.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
