"""Azure-only true-resolution perception probe. No endpoint or camera policy."""
import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from metrics import contains, gate, iou, lift, size_bucket, summarize

HERE = Path(__file__).resolve().parent
DRONE = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--azure-vm', action='store_true', help='Confirm execution on Azure VM')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux' or not args.azure_vm:
        parser.error('Inference is allowed only on Azure Linux; pass --azure-vm there.')
    # Never inherit serving knobs from the frozen runtime image.
    cleared = sorted(name for name in os.environ if name.startswith('DRONE_'))
    for name in cleared:
        del os.environ[name]
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(DRONE))
    import numpy as np
    from local_evaluator import Camera, render_view
    from utils import (center_bounds_for_level, decode_image, frame_numbers,
                       load_annotations, load_frame, scene_directory)
    from v2.config import CONFIG
    from v2.pipeline import DroneFlybyPipeline
    from v2.proposals import OnnxYoloProposer, merge_proposals

    frames = frame_numbers('helsinki')
    if frames != list(range(25)):
        raise RuntimeError(f'Expected the supplied 25-frame sequence, got {frames}')
    data = scene_directory('helsinki')
    paths = [p for folder in ('images', 'annotations') for p in sorted((data / folder).iterdir())
             if p.is_file()]
    paths += [DRONE / 'models' / name for name in
              ('v3_oneclass_standard.onnx', 'v3_oneclass_standard.pt', 'reference_gallery.npz')]
    pc = replace(CONFIG.proposals, backend='yolo', discovery_backend='v3_standard', budget=8,
                 yolo_imgsz=960, yolo_confidence=.01, yolo_iou=.55,
                 yolo_weights=str(DRONE / 'models/v3_oneclass_standard.onnx'))
    config = replace(CONFIG, proposals=pc,
                     scheduler=replace(CONFIG.scheduler, working_level=0),
                     telemetry=replace(CONFIG.telemetry, enabled=False))
    for name, expected in (('v3_oneclass_standard.onnx', pc.v3_standard_onnx_sha256),
                           ('v3_oneclass_standard.pt', pc.v3_standard_pt_sha256)):
        if sha(DRONE / 'models' / name) != expected:
            raise RuntimeError(f'Frozen asset checksum mismatch: {name}')
    paths += sorted((Path(os.environ.get('TORCH_HOME', '/app/.torch')) / 'hub/checkpoints').glob('resnet18*'))
    manifest = dict(config=asdict(config), platform=platform.platform(), cleared_environment=cleared,
                    hashes={str(p): sha(p) for p in paths},
                    packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
                    methodology='oracle_target_centered; independent static crops; not a legal trajectory')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    class FailOnModelError(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                raise RuntimeError(record.getMessage())
    for module in ('v2.proposals', 'v2.recognizer'):
        logging.getLogger(module).addHandler(FailOnModelError())
    pipeline = DroneFlybyPipeline(config)
    # Strict direct proposer avoids ProposalEngine's exception swallowing/fallback.
    if len(pipeline.proposals.backends) != 1 or not isinstance(pipeline.proposals.backends[0], OnnxYoloProposer):
        raise RuntimeError('Require frozen ONNX backend; no fallback permitted')
    detector = pipeline.proposals.backends[0]
    pipeline.load()
    if not pipeline.recognizer.available:
        raise RuntimeError(f'Recognizer unavailable: {pipeline.recognizer.load_error}')
    blank = np.zeros((540, 960, 3), dtype=np.uint8)
    for level in (0, 1, 2):
        detector.propose(blank, 300, level)
    pipeline.recognizer.warmup()
    objects, views = [], []
    for frame in frames:
        source = load_frame(frame, 'helsinki')
        if source.shape != (2160, 3840, 3):
            raise RuntimeError('Must use original 3840x2160 frames, never archived L0 PNGs')
        annotations = load_annotations(frame, 'helsinki')
        # Multiple oracle targets can share a crop. Cache inference, never duplicate latency samples.
        cache = {}
        for index, ann in enumerate(annotations):
            box = ann['bbox']
            for level in (0, 1, 2):
                xmin, xmax, ymin, ymax = center_bounds_for_level(level)
                cx = min(xmax, max(xmin, round((box[0] + box[2]) / 2)))
                cy = min(ymax, max(ymin, round((box[1] + box[3]) / 2)))
                camera = Camera(level, cx, cy)
                region = camera.source_region
                if not contains(region, box):
                    raise RuntimeError('Oracle target not fully visible; paired denominator would change')
                key = (level, cx, cy)
                if key not in cache:
                    image = decode_image(render_view(source, camera))
                    cache[key] = {}
                    # Alternate A/B order across frames to reduce systematic warm-cache bias.
                    modes = ('matched', 'native') if frame % 2 == 0 else ('native', 'matched')
                    for mode in modes:
                        inference_level = level if mode == 'matched' else 0
                        started = time.perf_counter()
                        raw = detector.propose(image, 300, inference_level)
                        ranked = merge_proposals(raw, pc.merge_iou, 300)
                        discovery_ms = (time.perf_counter() - started) * 1000
                        cache[key][mode] = {}
                        for budget in (8, 32):
                            proposals = ranked[:budget]
                            started = time.perf_counter()
                            observations, rec = pipeline._recognize(image, proposals, region, level)
                            recognition_ms = (time.perf_counter() - started) * 1000
                            boxes = [lift(p.box, region) for p in proposals]
                            cache[key][mode][budget] = boxes
                            # All GT, including partial objects, prevents counting crop-edge objects as terrain.
                            overlaps = [max((iou(b, a['bbox']) for a in annotations), default=0)
                                        for b in boxes]
                            views.append(dict(frame=frame, level=level, center=[cx, cy], mode=mode,
                                budget=budget, requested_imgsz=960 >> inference_level,
                                detector_canvas=list(detector._letterbox(image, 960 >> inference_level)[0].shape[:2]),
                                count=len(boxes), background_iou_lt_01=sum(v < .1 for v in overlaps),
                                unmatched_iou_lt_05=sum(v < .5 for v in overlaps),
                                proposals=[dict(box=b, score=p.score) for b, p in zip(boxes, proposals)],
                                recognition_rejected=rec.get('rejected_as_background', 0),
                                observations=len(observations), discovery_ms=discovery_ms,
                                recognition_ms=recognition_ms, total_ms=discovery_ms + recognition_ms))
                for mode in ('matched', 'native'):
                    for budget in (8, 32):
                        best = max((iou(box, b) for b in cache[key][mode][budget]), default=0)
                        objects.append(dict(frame=frame, object_index=index, object_class=ann['object_id'],
                            gt_box=box, level=level, mode=mode, budget=budget,
                            size_source_short_side=size_bucket(box), best_iou=best))
        print(f'Phase A: frame {frame + 1}/25, {len(views)} measured configurations', flush=True)
        # Partial results survive an interrupted run; only the final summary carries a decision.
        for filename, rows in (('oracle_objects.json', objects), ('oracle_views.json', views)):
            (args.out / filename).write_text(json.dumps(rows, indent=2))
    latency = []
    for mode in ('matched', 'native'):
        for level in (0, 1, 2):
            for budget in (8, 32):
                subset = [r for r in views if (r['mode'], r['level'], r['budget']) == (mode, level, budget)]
                latency.append(dict(mode=mode, level=level, budget=budget, views=len(subset),
                    **{f'{stage}_p{q}': float(np.percentile([r[stage] for r in subset], q))
                       for stage in ('discovery_ms', 'recognition_ms', 'total_ms') for q in (50, 95)},
                    background_per_view=float(np.mean([r['background_iou_lt_01'] for r in subset])),
                    unmatched_per_view=float(np.mean([r['unmatched_iou_lt_05'] for r in subset]))))
    result = dict(gate=gate(objects), proposal_metrics=summarize(objects), latency_and_burden=latency,
                  warning='Oracle crop selection overestimates achievable camera coverage; no hosted claim.')
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['gate'], indent=2))


if __name__ == '__main__':
    main()
