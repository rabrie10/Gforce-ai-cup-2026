"""Probe D - can real pixels estimate the affine the GT oracle selected?

D-007 chose affine as the first image-derived GMC target on GT-oracle evidence:
strict one-step held-out success 238/243 = 97.94%, non-boundary 220/220. That
answered "which transform class is worth estimating". It did not answer "can
correspondences on transmitted 960x540 images estimate it well enough to
propagate a track the camera is not currently looking at".

This probe answers the second question with the same yardstick the oracle used,
so the two numbers are directly comparable: take a GT box at frame f, push it
through the *image-derived* transforms for h steps, and ask whether it still
overlaps the GT box at frame f+h at IoU >= 0.50.

Camera policies matter here and are swept deliberately. At L0 the estimator
sees the whole frame and the transform is genuinely global. At L1 and L2 it
sees 25% and 6.25%, so applying the estimate to a track outside the crop is an
extrapolation - which is exactly what the V2 pipeline will be doing, and
exactly what needs measuring before it is trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtos import MAXIMUM_CENTER_DELTA_PIXELS  # noqa: E402
from utils import frame_numbers, load_annotations, load_frame  # noqa: E402
from v2.config import GmcConfig  # noqa: E402
from v2.geometry import (  # noqa: E402
    IDENTITY_AFFINE,
    apply_affine_to_box,
    box_center,
    compose_affine,
    contains,
    iou,
    render_level_view,
    source_region,
)
from v2.gmc import GlobalMotionEstimator  # noqa: E402


HORIZONS = (1, 2, 3, 4, 5, 6)


def camera_track(policy: str, frames: Sequence[int]) -> List[Tuple[int, int, int]]:
    """A legal (level, cx, cy) per frame for the policy under test."""
    if policy == 'l0':
        return [(0, 1920, 1080) for _ in frames]
    if policy == 'l1_static':
        return [(1, 1920, 1080) for _ in frames]
    if policy == 'l2_static':
        return [(2, 1920, 1080) for _ in frames]
    if policy == 'l1_sweep':
        # Alternate between two L1 centres 960 px apart, inside the 1102 px
        # per-frame limit, so the estimator has to cope with a moving camera.
        track = []
        centres = [(1440, 1080), (2400, 1080)]
        for index in range(len(frames)):
            track.append((1,) + centres[index % 2])
        limit = MAXIMUM_CENTER_DELTA_PIXELS[1]
        for previous, current in zip(track, track[1:]):
            distance = np.hypot(current[1] - previous[1], current[2] - previous[2])
            assert distance <= limit, f'illegal sweep step {distance}'
        return track
    raise ValueError(f'unknown policy {policy!r}')


def run_policy(policy: str, scene: str, model: str) -> dict:
    frames = frame_numbers(scene)
    track = camera_track(policy, frames)
    estimator = GlobalMotionEstimator(GmcConfig(model=model))

    transforms: List[Optional[np.ndarray]] = [None]
    estimates = []
    annotations_by_frame = {}

    for index, frame in enumerate(frames):
        image = load_frame(frame, scene)
        level, center_x, center_y = track[index]
        region = source_region(level, center_x, center_y)
        view = render_level_view(image, level, center_x, center_y)
        annotations_by_frame[frame] = [
            (row['object_id'], tuple(float(v) for v in row['bbox']))
            for row in load_annotations(frame, scene)
        ]
        estimate = estimator.estimate(view, region)
        estimates.append(estimate)
        if index > 0:
            transforms.append(estimate.matrix if estimate.ok else None)

    # Propagate GT boxes through composed image-derived transforms.
    results = {horizon: {'valid': 0, 'eligible': 0, 'errors': []} for horizon in HORIZONS}
    hold = {horizon: 0 for horizon in HORIZONS}
    in_view = {horizon: {'valid': 0, 'eligible': 0} for horizon in HORIZONS}
    off_view = {horizon: {'valid': 0, 'eligible': 0} for horizon in HORIZONS}

    for start_index, frame in enumerate(frames):
        level, center_x, center_y = track[start_index]
        region = source_region(level, center_x, center_y)
        for object_id, box in annotations_by_frame[frame]:
            visible = contains(region, box)
            for horizon in HORIZONS:
                end_index = start_index + horizon
                if end_index >= len(frames):
                    continue
                target = next(
                    (
                        candidate
                        for name, candidate in annotations_by_frame[frames[end_index]]
                        if name == object_id
                    ),
                    None,
                )
                if target is None:
                    continue

                chain = transforms[start_index + 1:end_index + 1]
                if any(matrix is None for matrix in chain):
                    # A rejected transform means "no propagation available",
                    # which is a real cost of quality gating and is counted as
                    # an eligible failure rather than quietly skipped.
                    results[horizon]['eligible'] += 1
                    (in_view if visible else off_view)[horizon]['eligible'] += 1
                    continue

                composed = IDENTITY_AFFINE.copy()
                for matrix in chain:
                    composed = compose_affine(matrix, composed)
                predicted = apply_affine_to_box(composed, box)

                results[horizon]['eligible'] += 1
                bucket = in_view if visible else off_view
                bucket[horizon]['eligible'] += 1
                if iou(predicted, target) >= 0.50:
                    results[horizon]['valid'] += 1
                    bucket[horizon]['valid'] += 1
                predicted_center = box_center(predicted)
                target_center = box_center(target)
                results[horizon]['errors'].append(
                    float(np.hypot(
                        predicted_center[0] - target_center[0],
                        predicted_center[1] - target_center[1],
                    ))
                )
                if iou(box, target) >= 0.50:
                    hold[horizon] += 1

    accepted = sum(1 for estimate in estimates[1:] if estimate.ok)
    reasons: Dict[str, int] = defaultdict(int)
    for estimate in estimates[1:]:
        reasons[estimate.reason if not estimate.ok else f'ok:{estimate.model}'] += 1

    summary = {
        'policy': policy,
        'requested_model': model,
        'frames': len(frames),
        'transform_attempts': len(estimates) - 1,
        'transform_accepted': accepted,
        'transform_accept_rate': accepted / max(1, len(estimates) - 1),
        'outcomes': dict(reasons),
        'latency_ms_mean': float(np.mean([e.elapsed_ms for e in estimates])),
        'latency_ms_p95': float(np.percentile([e.elapsed_ms for e in estimates], 95)),
        'inlier_ratio_mean': float(
            np.mean([e.inlier_ratio for e in estimates[1:] if e.ok] or [0.0])
        ),
        'horizons': {},
    }
    for horizon in HORIZONS:
        eligible = results[horizon]['eligible']
        errors = results[horizon]['errors']
        summary['horizons'][str(horizon)] = {
            'valid': results[horizon]['valid'],
            'eligible': eligible,
            'success_rate': results[horizon]['valid'] / eligible if eligible else 0.0,
            'hold_valid': hold[horizon],
            'median_center_error_px': float(np.median(errors)) if errors else None,
            'p90_center_error_px': float(np.percentile(errors, 90)) if errors else None,
            'in_view': {
                'valid': in_view[horizon]['valid'],
                'eligible': in_view[horizon]['eligible'],
                'success_rate': (
                    in_view[horizon]['valid'] / in_view[horizon]['eligible']
                    if in_view[horizon]['eligible'] else None
                ),
            },
            'off_view': {
                'valid': off_view[horizon]['valid'],
                'eligible': off_view[horizon]['eligible'],
                'success_rate': (
                    off_view[horizon]['valid'] / off_view[horizon]['eligible']
                    if off_view[horizon]['eligible'] else None
                ),
            },
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--policies', default='l0,l1_static,l1_sweep,l2_static')
    parser.add_argument('--models', default='affine,similarity')
    parser.add_argument('--output', default=str(Path(__file__).parent / 'exp_d_results.json'))
    arguments = parser.parse_args()

    report = {
        'protocol': {
            'scene': arguments.scene,
            'horizons': list(HORIZONS),
            'comparator': (
                'GT-oracle strict LOO affine reached 238/243 = 0.9794 at h=1. '
                'A rejected transform counts as an eligible failure here, so '
                'these rates include the cost of quality gating.'
            ),
        },
        'runs': {},
    }

    for model in [value.strip() for value in arguments.models.split(',') if value.strip()]:
        for policy in [value.strip() for value in arguments.policies.split(',') if value.strip()]:
            key = f'{model}:{policy}'
            print(f'\n=== {key} ===')
            summary = run_policy(policy, arguments.scene, model)
            report['runs'][key] = summary
            print(
                f'  accepted {summary["transform_accepted"]}/{summary["transform_attempts"]}'
                f'  {summary["latency_ms_mean"]:.1f} ms mean  '
                f'inlier ratio {summary["inlier_ratio_mean"]:.3f}'
            )
            for horizon in HORIZONS:
                row = summary['horizons'][str(horizon)]
                error = row['median_center_error_px']
                print(
                    f'    h={horizon}: {row["valid"]:4d}/{row["eligible"]:4d} '
                    f'= {row["success_rate"]:.4f}   hold {row["hold_valid"]:4d}   '
                    f'median centre err {error if error is None else round(error, 2)} px'
                )

    Path(arguments.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nWrote {arguments.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
