"""Replay Helsinki through the pipeline in-process and attribute the score.

The local evaluator says what the score is. This says why: how many proposals
were made, how many of them found a real object, what the recognizer said about
the ones that did, how many tracks the bank is carrying, and how many of those
tracks correspond to nothing at all.

It drives the same camera state machine the evaluator does, so the camera
trajectory and the skipped-frame behaviour are the real ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dtos import (  # noqa: E402
    ALLOWED_RESOLUTION_LEVELS,
    FULL_FRAME_CENTER,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAXIMUM_CENTER_DELTA_PIXELS,
    OBJECT_CLASSES,
    SOURCE_REGION_SIZES,
    CameraConstraintsDto,
    CameraLevelBoundsDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyViewDto,
)
from utils import encode_image, frame_numbers, load_annotations, load_frame  # noqa: E402
from v2.geometry import (  # noqa: E402
    box_center,
    contains,
    iou,
    render_level_view,
    source_region,
)
from v2.pipeline import DroneFlybyPipeline  # noqa: E402


def constraints_for(level: int) -> CameraConstraintsDto:
    allowed = ALLOWED_RESOLUTION_LEVELS[level]
    bounds = []
    for target in allowed:
        width, height = SOURCE_REGION_SIZES[target]
        bounds.append(
            CameraLevelBoundsDto(
                resolution_level=target, width=960, height=540,
                minimum_center_x=width // 2, maximum_center_x=IMAGE_WIDTH - width // 2,
                minimum_center_y=height // 2, maximum_center_y=IMAGE_HEIGHT - height // 2,
            )
        )
    return CameraConstraintsDto(
        maximum_center_delta=MAXIMUM_CENTER_DELTA_PIXELS[level],
        allowed_resolution_levels=list(allowed),
        center_bounds=bounds,
        full_view_reset_exempt_from_delta=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='helsinki')
    parser.add_argument('--output', default=str(ROOT / 'architecture_experiments'
                                                / 'v2_component_probes' / 'diagnosis.json'))
    arguments = parser.parse_args()

    pipeline = DroneFlybyPipeline()
    pipeline.warmup()

    level, center = 0, FULL_FRAME_CENTER
    rows: List[dict] = []
    identity_hits, identity_total = 0, 0
    confusion: Counter = Counter()
    per_class_seen: Dict[str, int] = defaultdict(int)
    per_class_named: Dict[str, int] = defaultdict(int)

    for index, frame in enumerate(frame_numbers(arguments.scene)):
        image = load_frame(frame, arguments.scene)
        region = source_region(level, *center)
        view = render_level_view(image, level, *center)
        annotations = [
            (row['object_id'], tuple(float(v) for v in row['bbox']))
            for row in load_annotations(frame, arguments.scene)
        ]
        visible = [(name, box) for name, box in annotations if contains(region, box)]

        request = DroneFlybyPredictRequestDto(
            sequence_id='diagnose', frame=frame, frame_index=index,
            request_id=f'diagnose-{index}', frame_interval_ms=333, response_timeout_ms=3333,
            original_width=IMAGE_WIDTH, original_height=IMAGE_HEIGHT,
            view=DroneFlybyViewDto(
                resolution_level=level, center_x=center[0], center_y=center[1],
                view_id=f'v{index}', image=encode_image(view), image_media_type='image/png',
                width=960, height=540, source_region_xyxy=[int(v) for v in region],
            ),
            camera_constraints=constraints_for(level),
        )
        response = pipeline.predict(request)

        # Proposal recall, measured in source coordinates against what was visible.
        observations = getattr(pipeline, '_last_observations', [])
        found = 0
        for name, box in visible:
            centre = box_center(box)
            for observation in observations:
                if (observation.box[0] <= centre[0] <= observation.box[2]
                        and observation.box[1] <= centre[1] <= observation.box[3]):
                    found += 1
                    identity_total += 1
                    per_class_seen[name] += 1
                    predicted = OBJECT_CLASSES[int(np.argmax(observation.posterior))]
                    confusion[(name, predicted)] += 1
                    if predicted == name:
                        identity_hits += 1
                        per_class_named[name] += 1
                    break

        # Track quality: does each track sit on a real object?
        matched_tracks = 0
        for track in pipeline.bank.tracks:
            if any(iou(track.box, box) >= 0.50 for _, box in annotations):
                matched_tracks += 1

        emitted = Counter(a.object_id for a in response.annotations)
        rows.append({
            'frame': frame,
            'level': level,
            'center': list(center),
            'gt_total': len(annotations),
            'gt_visible': len(visible),
            'observations': len(observations),
            'proposal_center_hits': found,
            'tracks': len(pipeline.bank.tracks),
            'tracks_on_real_objects': matched_tracks,
            'annotations': len(response.annotations),
            'distinct_classes_emitted': len(emitted),
        })
        print(
            f'f{frame:3d} L{level} ({center[0]:4d},{center[1]:4d})  '
            f'gt {len(annotations):2d} vis {len(visible):2d}  '
            f'obs {len(observations):3d}  hit {found:2d}  '
            f'tracks {len(pipeline.bank.tracks):3d} ({matched_tracks:2d} real)  '
            f'ann {len(response.annotations):3d} / {len(emitted):2d} classes'
        )

        if response.requested_view is not None:
            level = response.requested_view.resolution_level
            center = (response.requested_view.center_x, response.requested_view.center_y)

    total_visible = sum(row['gt_visible'] for row in rows)
    total_hits = sum(row['proposal_center_hits'] for row in rows)
    summary = {
        'frames': len(rows),
        'proposal_center_recall': total_hits / max(1, total_visible),
        'gt_visible_total': total_visible,
        'mean_observations': float(np.mean([row['observations'] for row in rows])),
        'mean_tracks': float(np.mean([row['tracks'] for row in rows])),
        'mean_tracks_on_real_objects': float(
            np.mean([row['tracks_on_real_objects'] for row in rows])
        ),
        'track_precision': float(
            np.sum([row['tracks_on_real_objects'] for row in rows])
            / max(1, np.sum([row['tracks'] for row in rows]))
        ),
        'mean_annotations': float(np.mean([row['annotations'] for row in rows])),
        'recognizer_top1_on_hit_proposals': identity_hits / max(1, identity_total),
        'recognizer_evaluated': identity_total,
        'per_class': {
            name: {'seen': per_class_seen[name], 'named': per_class_named[name]}
            for name in sorted(per_class_seen)
        },
        'top_confusions': [
            {'gt': gt, 'predicted': predicted, 'n': count}
            for (gt, predicted), count in confusion.most_common(15)
        ],
        'latency_ms': pipeline.latency.report(),
    }
    print('\n' + json.dumps(summary, indent=2))
    Path(arguments.output).write_text(
        json.dumps({'summary': summary, 'frames': rows}, indent=2), encoding='utf-8'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
