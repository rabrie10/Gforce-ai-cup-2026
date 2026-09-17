"""Run from repository root: python drone-flyby/evaluator_probes/run_all.py."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CHALLENGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHALLENGE))

import local_evaluator as evaluator
from dtos import DroneFlybyPredictResponseDto
from evaluator_probes.harness import detection, iou, make_scene, run_replay, score
from evaluator_probes.diagnostics import precision_controls, timing_controls


def git(*args):
    return subprocess.check_output(['git', '-C', str(CHALLENGE.parent), *args], text=True).strip()


def fingerprints():
    files = [CHALLENGE / name for name in ('local_evaluator.py', 'dtos.py', 'utils.py', 'api.py', 'example.py')]
    files += sorted((CHALLENGE / 'src' / 'helsinki').rglob('*'))
    return {str(p.relative_to(CHALLENGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def measured_recall(scene, predictions):
    # Used only for exact-oracle subsets: counting retained GT is independent of COCO.
    oracle = evaluator.oracle_predictions(scene)
    total = Counter(d['object_id'] for ds in oracle.values() for d in ds)
    kept = Counter(d['object_id'] for ds in predictions.values() for d in ds)
    return dict(total_gt=sum(total.values()), retained_gt=sum(kept.values()),
                micro_recall=sum(kept.values()) / sum(total.values()),
                recall_by_class={c: kept[c] / n for c, n in total.items()},
                gt_by_class=dict(total), retained_by_class=dict(kept))


def p0(root):
    results = {}
    one = make_scene(root, 'single')
    sequence = make_scene(root, 'crop', 3)
    crop = {}
    for name, center in [('in_crop', (960, 540)), ('off_crop', (2880, 1620))]:
        def move(body, payload, center=center):
            if payload['frame_index'] == 0:
                body['requested_view'] = dict(resolution_level=1, center_x=center[0], center_y=center[1])
        run = run_replay(sequence, move)
        for request in run['requests']:
            box = (100, 100, 300, 300)
            region = request['view']['source_region_xyxy']
            request['gt_intersects_crop'] = (min(box[2], region[2]) > max(box[0], region[0])
                                              and min(box[3], region[3]) > max(box[1], region[1]))
        crop[name] = run
    results['A'] = crop
    results['B'] = []
    for target in (0.499, 0.500, 0.501, 0.60, 0.90):
        # Containment gives IoU exactly width_ratio; 0.500 uses integer corners.
        gt = [100, 100, 300, 300]
        box = [100, 100, 100 + 200 * target, 300]
        results['B'].append(dict(target_iou=target, computed_iou=iou(gt, box), gt=gt, prediction=box,
                                 **score(one, {0: [detection(box)]})))
    tp, fp = detection(), detection((600, 600, 800, 800), 0.1)
    results['C'] = {}
    for name, tp_conf, fp_conf in [('TP_first', .9, .1), ('FP_first', .1, .9),
                                  ('TP_first_squared', .81, .01), ('FP_first_squared', .01, .81)]:
        predictions = [dict(tp, confidence=tp_conf), dict(fp, confidence=fp_conf)]
        results['C'][name] = dict(predictions=predictions, **score(one, {0: predictions}))
    results['D'] = {}
    duplicates = {
        'one': [tp],
        'duplicate_below': [tp, dict(tp, confidence=.1)],
        'duplicate_above': [dict(tp, confidence=.1), tp],
        'five_duplicates_below': [tp] + [dict(tp, confidence=.5 - i * .05) for i in range(5)],
    }
    for name, predictions in duplicates.items():
        results['D'][name] = score(one, {0: predictions})
    two = make_scene(root, 'two_objects', objects=[tp, detection((600, 600, 800, 800))])
    for name, duplicate_conf in [('duplicate_before_second_TP', .8), ('duplicate_after_all_TPs', .1)]:
        predictions = [tp, dict(tp, confidence=duplicate_conf), detection((600, 600, 800, 800), .5)]
        results['D'][name] = dict(predictions=predictions, **score(two, {0: predictions}))
    oracle = evaluator.oracle_predictions('helsinki')
    results['E'] = {}
    for name, remove in [('none', lambda f: False), ('frame_0', lambda f: f == 0),
                          ('every_second_even', lambda f: f % 2 == 0),
                          ('every_third_zero', lambda f: f % 3 == 0),
                          ('last_five', lambda f: f >= 20)]:
        predictions = {f: ds for f, ds in oracle.items() if not remove(f)}
        results['E'][name] = dict(removed_frames=[f for f in oracle if remove(f)],
                                  **score('helsinki', predictions), **measured_recall('helsinki', predictions))
    results['F'] = []
    for delay in (0, 250, 300, 320, 330, 333, 340, 350, 400, 500, 650, 700, 1000):
        print(f'P0 F: Helsinki realtime, endpoint sleep {delay} ms', flush=True)
        run = run_replay('helsinki', delay_ms=delay, realtime=True)
        run.update(measured_recall('helsinki', {f: oracle[f] for f in run['predicted_frames']}))
        results['F'].append(run)
    results['G'] = []
    for rank in (99, 100, 101):
        predictions = [detection((600, 600, 800, 800), 1 - i / 200) for i in range(101)]
        predictions[rank - 1]['bbox'] = tp['bbox']
        results['G'].append(dict(tp_rank=rank, detection_count=len(predictions), **score(one, {0: predictions})))
    from faster_coco_eval import COCOeval_faster
    params = COCOeval_faster(iouType='bbox').params
    results['G_configuration'] = dict(maxDets=params.maxDets, recall_thresholds=params.recThrs.tolist())
    return results


def p1(root):
    results = {}
    scene = make_scene(root, 'invalid', 2, [detection(), detection((600, 600, 800, 800), object_id='tank')])
    cases = ('valid_control', 'bbox_reversed', 'bbox_zero_width', 'bbox_out_of_range', 'nan_string',
             'infinity_string', 'infinity_numeric_overflow', 'invalid_class', 'unexpected_key',
             'wrong_request_id', 'wrong_frame', '501_annotations', 'float_camera', 'illegal_geometry')
    results['H'] = {}
    for case in cases:
        def mutate(body, payload, case=case):
            if payload['frame_index'] != 0:
                return
            body['requested_view'] = dict(resolution_level=1, center_x=960, center_y=540)
            bad = copy.deepcopy(body['annotations'][0])
            # Preserve both good detections; add one isolated invalid annotation.
            if case.startswith('bbox_') or case.startswith('nan_') or case.startswith('infinity_') or case == 'invalid_class':
                body['annotations'].append(bad)
            if case == 'bbox_reversed': bad['bbox'] = [.3, .1, .2, .2]
            elif case == 'bbox_zero_width': bad['bbox'] = [.1, .1, .1, .2]
            elif case == 'bbox_out_of_range': bad['bbox'] = [-.01, .1, .2, .2]
            elif case == 'nan_string': bad['bbox'][0] = 'NaN'
            elif case == 'infinity_string': bad['bbox'][0] = 'Infinity'
            elif case == 'infinity_numeric_overflow': bad['bbox'][0] = 'OVERFLOW_NUMBER'
            elif case == 'invalid_class': bad['object_id'] = 'not_a_class'
            elif case == 'unexpected_key': body['debug'] = True
            elif case == 'wrong_request_id': body['request_id'] += '-wrong'
            elif case == 'wrong_frame': body['frame'] += 1
            elif case == '501_annotations': body['annotations'] = [bad] * 501
            elif case == 'float_camera': body['requested_view']['center_x'] = 960.0
            elif case == 'illegal_geometry': body['requested_view']['center_x'] = 0
        results['H'][case] = run_replay(scene, mutate)
    # IEEE NaN/Inf cannot be emitted by a strict JSON encoder. Probe the DTO
    # directly as well, without manufacturing a nonstandard NaN JSON token.
    results['H_nonfinite_DTO'] = []
    for name, value in [('NaN', float('nan')), ('Infinity', float('inf'))]:
        body = dict(request_id='probe', frame=0, annotations=[dict(object_id='hangar', bbox=[value, .1, .2, .2], confidence=.9)])
        try:
            json.dumps(body, allow_nan=False)
            wire_error = None
        except ValueError as exc:
            wire_error = str(exc)
        try:
            DroneFlybyPredictResponseDto.model_validate(body)
            valid, error = True, None
        except Exception as exc:
            valid, error = False, str(exc)
        results['H_nonfinite_DTO'].append(dict(value=name, schema_valid=valid, schema_error=error,
                                                strict_json_error=wire_error))
    results['I'] = {k: results['H'][k] for k in ('valid_control', 'illegal_geometry', 'float_camera')}
    feedback_scene = make_scene(root, 'feedback', 5)
    def feedback(body, payload):
        if payload['frame_index'] == 0:
            body['requested_view'] = dict(resolution_level=1, center_x=0, center_y=540)
        elif payload['frame_index'] == 3:
            body['requested_view'] = dict(resolution_level=1, center_x=960, center_y=540)
    results['J'] = run_replay(feedback_scene, feedback)
    results['K'] = []
    for delay in (3000, 3500):
        print(f'P1 K: Helsinki realtime, endpoint sleep {delay} ms', flush=True)
        results['K'].append(run_replay('helsinki', delay_ms=delay, realtime=True))
    return results


def absent(root):
    scene = make_scene(root, 'absent')
    return {name: score(scene, {0: predictions}) for name, predictions in {
        'oracle': [detection()],
        'oracle_plus_absent_class': [detection(), detection((600, 600, 800, 800), 1, 'tank')],
        'absent_class_only': [detection((600, 600, 800, 800), 1, 'tank')],
    }.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('probe_results.json'))
    args = parser.parse_args()
    hashes = fingerprints()
    env = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
               platform=platform.platform(), executable=sys.executable, commit=git('rev-parse', 'HEAD'),
               monotonic_clock=vars(time.get_clock_info('monotonic')),
               branch=git('branch', '--show-current'), git_status=git('status', '--short'),
               dependencies={name: importlib.metadata.version(name) for name in
                             ('faster-coco-eval', 'numpy', 'opencv-python', 'pydantic', 'requests', 'fastapi', 'uvicorn')},
               input_sha256=hashes, command=subprocess.list2cmdline(sys.argv))
    print('Phase 1: supplied oracle CLI', flush=True)
    oracle_cli = subprocess.run([sys.executable, str(CHALLENGE / 'local_evaluator.py'), '--oracle'],
                                capture_output=True, text=True, check=True)
    oracle = score('helsinki', evaluator.oracle_predictions('helsinki'))
    if abs(oracle['score'] - 1) > 1e-12 or any(abs(v - 1) > 1e-12 for v in oracle['ap_by_class'].values()):
        raise RuntimeError(f'Oracle failed; no further probes run: {oracle}')
    results = dict(environment=env, oracle=dict(**oracle, stdout=oracle_cli.stdout,
                                                command=f'{sys.executable} drone-flyby/local_evaluator.py --oracle'))
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='drone-evaluator-probes-') as root:
        results.update(p0(root))
        results['F_controls'] = timing_controls(root)
        results['C_precision'] = precision_controls(root)
        save()
        print('P0 complete. Starting robustness probes.', flush=True)
        results.update(p1(root))
        results['L_absent_class'] = absent(root)
    results['authoritative_inputs_unchanged'] = hashes == fingerprints()
    if not results['authoritative_inputs_unchanged']:
        raise RuntimeError('Authoritative input hash changed during run')
    save()
    from evaluator_probes.report import write_report
    report_path = (args.output.with_name('PROBE_RESULTS.md') if args.output.name == 'probe_results.json'
                   else args.output.with_suffix('.md'))
    write_report(results, report_path)
    print(f'Results: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
