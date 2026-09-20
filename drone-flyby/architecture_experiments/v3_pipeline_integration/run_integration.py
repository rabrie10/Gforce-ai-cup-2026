"""Reproduce the controlled V3 full-pipeline integration measurements.

The Helsinki path uses the repository evaluator's rendering, camera contract,
response validation and scorer. Hosted captures are intentionally stateless:
each sampled image is passed only through discovery and recognition.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DRONE = HERE.parents[1]
if str(DRONE) not in sys.path:
    sys.path.insert(0, str(DRONE))

from dtos import DroneFlybyPredictRequestDto, OBJECT_CLASSES  # noqa: E402
from local_evaluator import (  # noqa: E402
    Camera,
    CameraRejection,
    FRAME_INTERVAL_SECONDS,
    build_request,
    frame_numbers,
    load_frame,
    render_view,
    score,
)
from utils import global_bbox_to_source, validate_response  # noqa: E402
from v2.config import CONFIG  # noqa: E402
from v2.pipeline import DroneFlybyPipeline  # noqa: E402


V3_ONNX = DRONE / 'models' / 'v3_oneclass_standard.onnx'
V3_PT = DRONE / 'models' / 'v3_oneclass_standard.pt'
HOSTED = (
    DRONE / 'hosted_telemetry' / 'v2_validation_6a86911a'
    / 'run_20260919_005220' / 'images'
)
EXPECTED = {
    V3_ONNX: '2daeabaeee41ffca9285485920238ce09c7762423d2d4fc41e406625d1b44d8f',
    V3_PT: '126aa31373775b324cf0df37c683b983202f9bc4dbfafd6105f40d98fafcec07',
}
BUDGETS = (8, 16, 24, 32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def make_config(backend: str, budget: int):
    proposals = replace(
        CONFIG.proposals,
        backend='yolo',
        discovery_backend=backend,
        budget=budget,
        v3_standard_weights=str(V3_ONNX),
        v3_standard_fallback_weights=str(V3_PT),
    )
    return replace(CONFIG, proposals=proposals)


def run_helsinki(backend: str, budget: int, realtime: bool) -> tuple[dict, list[dict]]:
    pipeline = DroneFlybyPipeline(make_config(backend, budget))
    pipeline.warmup()
    frames = frame_numbers('helsinki')
    camera = Camera()
    predictions = {}
    diagnostics = []
    frame_index = 0
    started = time.monotonic()
    invalid_responses = invalid_commands = 0
    accepted = 0

    while frame_index < len(frames):
        frame = frames[frame_index]
        image = load_frame(frame, 'helsinki')
        payload = build_request(frame, frame_index, camera, render_view(image, camera), None)
        request = DroneFlybyPredictRequestDto.model_validate(payload)
        response = pipeline.predict(request)
        try:
            validate_response(response)
            if response.request_id != request.request_id or response.frame != frame:
                raise ValueError('request/frame echo mismatch')
        except Exception:
            invalid_responses += 1
            response = None

        if response is not None:
            accepted += 1
            predictions[frame] = [
                {
                    'object_id': annotation.object_id,
                    'bbox': global_bbox_to_source(
                        annotation.bbox, request.original_width, request.original_height
                    ),
                    'confidence': float(annotation.confidence),
                }
                for annotation in response.annotations
            ]
            if response.requested_view is not None:
                requested = response.requested_view
                try:
                    camera.apply(
                        requested.resolution_level, requested.center_x, requested.center_y
                    )
                except CameraRejection:
                    invalid_commands += 1

        row = dict(pipeline.last_diagnostics)
        row.update({'frame': frame, 'frame_index': frame_index})
        diagnostics.append(row)
        if realtime:
            elapsed = time.monotonic() - started
            frame_index = max(frame_index + 1, int(elapsed / FRAME_INTERVAL_SECONDS))
        else:
            frame_index += 1

    map50, per_class = score('helsinki', predictions)
    timings = [row['timing_ms'] for row in diagnostics]
    stages = sorted({name for row in timings for name in row})
    latency = {
        name: {
            'p50_ms': percentile([row.get(name, 0.0) for row in timings], 0.50),
            'p95_ms': percentile([row.get(name, 0.0) for row in timings], 0.95),
            'max_ms': max((row.get(name, 0.0) for row in timings), default=0.0),
        }
        for name in stages
    }
    emitted = [name for row in diagnostics for name in row['emitted_classes']]
    bank_sizes = [row['track_bank_size'] for row in diagnostics]
    created = sum(int(row.get('bank', {}).get('created', 0)) for row in diagnostics)
    removed = sum(int(row.get('bank', {}).get('removed', 0)) for row in diagnostics)
    result = {
        'backend': backend,
        'budget': budget,
        'harness': 'realtime' if realtime else 'offline-complete',
        'map50': map50,
        'per_class_ap': per_class,
        'frames_total': len(frames),
        'accepted_frames': accepted,
        'skipped_frames': len(frames) - len(diagnostics),
        'invalid_responses': invalid_responses,
        'invalid_commands': invalid_commands,
        'annotations_per_accepted_frame': len(emitted) / accepted if accepted else 0.0,
        'distinct_emitted_classes': sorted(set(emitted)),
        'emitted_class_distribution': dict(sorted(Counter(emitted).items())),
        'track_bank_mean': statistics.mean(bank_sizes) if bank_sizes else 0.0,
        'track_bank_max': max(bank_sizes, default=0),
        'track_bank_saturation_frames': sum(bool(row['track_bank_saturated']) for row in diagnostics),
        'tracks_created': created,
        'tracks_removed': removed,
        'candidate_to_output_conversion': (
            len(emitted) / sum(row['proposals']['after_budget'] for row in diagnostics)
            if diagnostics else 0.0
        ),
        'latency_ms': latency,
    }
    return result, diagnostics


def run_hosted(backend: str, budget: int) -> list[dict]:
    pipeline = DroneFlybyPipeline(make_config(backend, budget))
    pipeline.load()
    rows = []
    for path in sorted(HOSTED.glob('*.png')):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        started = time.perf_counter()
        proposals = pipeline.proposals.propose(image, level=0)
        proposal_ms = (time.perf_counter() - started) * 1000.0
        stats = dict(pipeline.proposals.last_stats)
        started = time.perf_counter()
        observations, recognition = pipeline._recognize(
            image, proposals, (0.0, 0.0, 3840.0, 2160.0), 0
        )
        recognize_ms = (time.perf_counter() - started) * 1000.0
        top_classes = recognition.get('top_classes', [])
        top_posteriors = recognition.get('top_posteriors', [])
        objectness = recognition.get('objectness', [])
        rows.append({
            'backend': backend,
            'budget': budget,
            'image': path.name,
            'sample_stride': 5,
            'state_reset': True,
            'proposals_before_budget': stats['before_budget'],
            'proposals_after_budget': stats['after_budget'],
            'proposal_score_p10': percentile(stats['scores'], 0.10),
            'proposal_score_p50': percentile(stats['scores'], 0.50),
            'proposal_score_p90': percentile(stats['scores'], 0.90),
            'recognition_crops': recognition.get('crops', 0),
            'rejected_as_background': recognition.get('rejected_as_background', 0),
            'rejection_fraction': (
                recognition.get('rejected_as_background', 0) / recognition.get('crops', 1)
                if recognition.get('crops', 0) else 0.0
            ),
            'observations': len(observations),
            'top_class_distribution': json.dumps(dict(sorted(Counter(top_classes).items()))),
            'top_posterior_p50': percentile(top_posteriors, 0.50),
            'top_posterior_p90': percentile(top_posteriors, 0.90),
            'objectness_p50': percentile(objectness, 0.50),
            'proposal_ms': proposal_ms,
            'recognition_ms': recognize_ms,
            'discovery_recognition_ms': proposal_ms + recognize_ms,
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    hashes = {str(path): sha256(path) for path in EXPECTED}
    for path, expected in EXPECTED.items():
        if hashes[str(path)] != expected:
            raise RuntimeError(f'hash mismatch for {path}')

    configs = [('v2', 32)] + [('v3_standard', budget) for budget in BUDGETS]
    helsinki = []
    for backend, budget in configs:
        print(f'Helsinki offline: {backend} K={budget}', flush=True)
        result, _ = run_helsinki(backend, budget, realtime=False)
        print(f'Helsinki realtime: {backend} K={budget}', flush=True)
        realtime_result, _ = run_helsinki(backend, budget, realtime=True)
        result['realtime'] = {
            'accepted_frames': realtime_result['accepted_frames'],
            'skipped_frames': realtime_result['skipped_frames'],
            'invalid_responses': realtime_result['invalid_responses'],
            'invalid_commands': realtime_result['invalid_commands'],
            'map50': realtime_result['map50'],
            'latency_ms': realtime_result['latency_ms'],
        }
        helsinki.append(result)

    hosted_rows = []
    for backend, budget in configs:
        print(f'Hosted stateless: {backend} K={budget}', flush=True)
        hosted_rows.extend(run_hosted(backend, budget))

    helsinki_rows = []
    for result in helsinki:
        for name in OBJECT_CLASSES:
            helsinki_rows.append({
                'backend': result['backend'],
                'budget': result['budget'],
                'map50': result['map50'],
                'class': name,
                'class_ap50': result['per_class_ap'].get(name, ''),
                'offline_accepted_frames': result['accepted_frames'],
                'realtime_accepted_frames': result['realtime']['accepted_frames'],
                'realtime_skipped_frames': result['realtime']['skipped_frames'],
                'invalid_responses': result['realtime']['invalid_responses'],
                'invalid_commands': result['realtime']['invalid_commands'],
                'annotations_per_accepted_frame': result['annotations_per_accepted_frame'],
                'distinct_classes': len(result['distinct_emitted_classes']),
                'track_bank_mean': result['track_bank_mean'],
                'track_bank_max': result['track_bank_max'],
            })
    write_csv(HERE / 'helsinki_full_pipeline_results.csv', helsinki_rows)
    write_csv(HERE / 'hosted_gt_free_pipeline_diagnostics.csv', hosted_rows)

    comparison = []
    for result in helsinki:
        subset = [
            row for row in hosted_rows
            if row['backend'] == result['backend'] and row['budget'] == result['budget']
        ]
        comparison.append({
            'backend': result['backend'],
            'budget': result['budget'],
            'helsinki_map50': result['map50'],
            'helsinki_offline_accepted_frames': result['accepted_frames'],
            'helsinki_realtime_accepted_frames': result['realtime']['accepted_frames'],
            'helsinki_realtime_skipped_frames': result['realtime']['skipped_frames'],
            'helsinki_realtime_map50': result['realtime']['map50'],
            'helsinki_annotations_per_frame': result['annotations_per_accepted_frame'],
            'helsinki_track_bank_mean': result['track_bank_mean'],
            'helsinki_track_bank_max': result['track_bank_max'],
            'helsinki_total_p95_ms': result['latency_ms']['total']['p95_ms'],
            'hosted_proposals_before_budget_mean': statistics.mean(
                row['proposals_before_budget'] for row in subset
            ),
            'hosted_proposals_after_budget_mean': statistics.mean(
                row['proposals_after_budget'] for row in subset
            ),
            'hosted_rejection_fraction_mean': statistics.mean(
                row['rejection_fraction'] for row in subset
            ),
            'hosted_observations_per_image': statistics.mean(row['observations'] for row in subset),
            'hosted_discovery_recognition_p95_ms': percentile(
                [row['discovery_recognition_ms'] for row in subset], 0.95
            ),
        })
    write_csv(HERE / 'proposal_budget_comparison.csv', comparison)

    latency = {
        f"{row['backend']}_k{row['budget']}": row['latency_ms'] for row in helsinki
    }
    (HERE / 'latency_results.json').write_text(
        json.dumps(latency, indent=2) + '\n', encoding='utf-8'
    )
    summary = {
        'experiment': 'V3 STAGE 2: CONTROLLED FULL-PIPELINE INTEGRATION',
        'v3_discovery_commit': '74d1e0181d794b425f06d13a625df3de2fb6f7dd',
        'frozen_v2_commit': '6fb764e5b9fb5f7eefcb832c72b35553165219e5',
        'model_hashes': hashes,
        'configs': [
            {
                'backend': backend,
                'budget': budget,
                'confidence_threshold': CONFIG.proposals.yolo_confidence,
                'nms_iou': CONFIG.proposals.yolo_iou,
            }
            for backend, budget in configs
        ],
        'helsinki': helsinki,
        'proposal_budget_comparison': comparison,
        'hosted_images': len(sorted(HOSTED.glob('*.png'))),
        'hosted_method': 'GT-free, stride-5 captures, independent discovery/recognition; no temporal metrics',
    }
    (HERE / 'integration_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
