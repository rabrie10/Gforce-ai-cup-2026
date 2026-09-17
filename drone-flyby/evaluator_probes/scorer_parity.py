"""Deterministic score-only comparison; never calls replay, HTTP, or camera APIs.

From repository root, install each scorer using --no-deps --target as documented
in README.md, then run this file using the existing probe environment's Python.
Each scorer is loaded in a separate child process with an asserted version/path.
"""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
CHALLENGE = DIRECTORY.parent
VERSIONS = ('1.7.2', '1.8.0')


def worker(version, package_dir, output):
    # Insert before importing any evaluator, helper, or compiled scorer module.
    sys.path[:0] = [str(package_dir.resolve()), str(CHALLENGE)]
    import faster_coco_eval
    import local_evaluator as evaluator
    from evaluator_probes.harness import detection, iou, make_scene, score
    from evaluator_probes.run_all import fingerprints, measured_recall
    from utils import global_bbox_to_source, source_bbox_to_global

    actual = importlib.metadata.version('faster-coco-eval')
    module_path = Path(faster_coco_eval.__file__).resolve()
    if actual != version or not module_path.is_relative_to(package_dir.resolve()):
        raise RuntimeError(f'Wrong scorer loaded: {actual}, {module_path}')
    before = fingerprints()
    cases = {}
    setup = {}
    oracle = evaluator.oracle_predictions('helsinki')
    cases['Oracle'] = score('helsinki', oracle)
    if cases['Oracle']['score'] != 1 or any(v != 1 for v in cases['Oracle']['ap_by_class'].values()):
        raise RuntimeError(f'Oracle failed: {cases["Oracle"]}')
    with tempfile.TemporaryDirectory(prefix=f'drone-score-{version}-') as root:
        one = make_scene(root, 'single')
        crop_scene = make_scene(root, 'crop', 3)
        crop_predictions = evaluator.oracle_predictions(crop_scene)
        for annotations in crop_predictions.values():
            for annotation in annotations:
                annotation['bbox'] = global_bbox_to_source(source_bbox_to_global(annotation['bbox']))
        for name, region in [('in_crop', [0, 0, 1920, 1080]), ('off_crop', [1920, 1080, 3840, 2160])]:
            cases[f'A/{name}'] = score(crop_scene, crop_predictions)
            setup[f'A/{name}'] = dict(source_regions=[[0, 0, 3840, 2160], region, region],
                                     predictions=crop_predictions,
                                     note='Crop metadata is not a score() input; no camera or replay calls.')
        for target in (.499, .500, .501, .60, .90):
            box = [100, 100, 100 + 200 * target, 300]
            name = f'B/{target:.3f}'
            cases[name] = score(one, {0: [detection(box)]})
            setup[name] = dict(target_iou=target, computed_iou=iou([100, 100, 300, 300], box), bbox=box)
        tp = detection()
        fp = detection((600, 600, 800, 800), .1)
        for name, tp_conf, fp_conf in [('TP_first', .9, .1), ('FP_first', .1, .9),
                                      ('TP_first_squared', .81, .01), ('FP_first_squared', .01, .81)]:
            predictions = [dict(tp, confidence=tp_conf), dict(fp, confidence=fp_conf)]
            cases[f'C/{name}'] = score(one, {0: predictions})
        duplicates = {'one': [tp], 'duplicate_below': [tp, dict(tp, confidence=.1)],
                      'duplicate_above': [dict(tp, confidence=.1), tp],
                      'five_duplicates_below': [tp] + [dict(tp, confidence=.5 - i * .05) for i in range(5)]}
        for name, predictions in duplicates.items():
            cases[f'D/{name}'] = score(one, {0: predictions})
        two = make_scene(root, 'two_objects', objects=[tp, detection((600, 600, 800, 800))])
        for name, confidence in [('duplicate_before_second_TP', .8), ('duplicate_after_all_TPs', .1)]:
            predictions = [tp, dict(tp, confidence=confidence), detection((600, 600, 800, 800), .5)]
            cases[f'D/{name}'] = score(two, {0: predictions})
        for name, remove in [('none', lambda f: False), ('frame_0', lambda f: f == 0),
                              ('every_second_even', lambda f: f % 2 == 0),
                              ('every_third_zero', lambda f: f % 3 == 0), ('last_five', lambda f: f >= 20)]:
            predictions = {f: ds for f, ds in oracle.items() if not remove(f)}
            cases[f'E/{name}'] = score('helsinki', predictions)
            setup[f'E/{name}'] = measured_recall('helsinki', predictions)
        for rank in (99, 100, 101):
            predictions = [detection((600, 600, 800, 800), 1 - i / 200) for i in range(101)]
            predictions[rank - 1]['bbox'] = tp['bbox']
            cases[f'G/{rank}'] = score(one, {0: predictions})
        for name, predictions in {
            'oracle': [tp],
            'oracle_plus_absent_class': [tp, detection((600, 600, 800, 800), 1, 'tank')],
            'absent_class_only': [detection((600, 600, 800, 800), 1, 'tank')],
        }.items():
            cases[f'L/{name}'] = score(one, {0: predictions})
    params = faster_coco_eval.COCOeval_faster(iouType='bbox').params
    unchanged = before == fingerprints()
    if not unchanged:
        raise RuntimeError('Authoritative inputs changed during scoring')
    result = dict(scorer_version=actual, scorer_module=str(module_path),
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                  platform=platform.platform(), executable=sys.executable,
                  dependencies={n: importlib.metadata.version(n) for n in ('numpy', 'opencv-python', 'pydantic', 'requests')},
                  input_sha256=before, authoritative_inputs_unchanged=unchanged, cases=cases,
                  setup=setup, configuration=dict(maxDets=params.maxDets, useCats=params.useCats,
                                                  recall_thresholds=params.recThrs.tolist()))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'{actual}: {len(cases)} deterministic scoring cases complete; loaded {module_path}', flush=True)


def differences(left, right, path=''):
    """Exact comparison without rounding or tolerance; enumerate every difference."""
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                rows.append(dict(path=f'{path}/{key}', left=left.get(key), right=right.get(key)))
            else:
                rows.extend(differences(left[key], right[key], f'{path}/{key}'))
        return rows
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [r for i, (a, b) in enumerate(zip(left, right)) for r in differences(a, b, f'{path}/{i}')]
    if left == right:
        return []
    row = dict(path=path, left=left, right=right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        row.update(delta=right - left, absolute_delta=abs(right - left),
                   left_hex=float(left).hex(), right_hex=float(right).hex())
    return [row]


def original_cases(data):
    keep = lambda result: {k: result[k] for k in ('score', 'ap_by_class')}
    cases = {'Oracle': keep(data['oracle'])}
    for probe in ('A', 'C', 'D', 'E'):
        cases.update({f'{probe}/{name}': keep(result) for name, result in data[probe].items()})
    cases.update({f'B/{r["target_iou"]:.3f}': keep(r) for r in data['B']})
    cases.update({f'G/{r["tp_rank"]}': keep(r) for r in data['G']})
    cases.update({f'L/{name}': keep(r) for name, r in data['L_absent_class'].items()})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-version', choices=VERSIONS)
    parser.add_argument('--package-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker_version:
        worker(args.worker_version, args.package_dir, args.output)
        return
    data_path = DIRECTORY / 'probe_results.json'
    data = json.loads(data_path.read_text(encoding='utf-8'))
    existing_version = importlib.metadata.version('faster-coco-eval')
    commands = []
    runs = {}
    for version in VERSIONS:
        relative_dir = f'drone-flyby/evaluator_probes/.cache/scorer-{version}'
        command = [sys.executable, 'drone-flyby/evaluator_probes/scorer_parity.py',
                   '--worker-version', version, '--package-dir', relative_dir,
                   '--output', f'drone-flyby/evaluator_probes/.cache/parity-{version}.json']
        commands.append(dict(version=version,
                             install=f'& drone-flyby/.venv/Scripts/python.exe -m pip install --no-deps --target {relative_dir} faster-coco-eval=={version}',
                             run_argv=command))
        subprocess.run(command, cwd=CHALLENGE.parent, check=True)
        runs[version] = json.loads((DIRECTORY / '.cache' / f'parity-{version}.json').read_text(encoding='utf-8'))
    left, right = (runs[v] for v in VERSIONS)
    metric_diffs = differences(left['cases'], right['cases'])
    config_diffs = differences(left['configuration'], right['configuration'])
    setup_diffs = differences(left['setup'], right['setup'])
    baseline_diffs = {v: differences(original_cases(data), runs[v]['cases']) for v in VERSIONS}
    checks = dict(other_dependencies_identical=left['dependencies'] == right['dependencies'],
                  authoritative_inputs_identical=left['input_sha256'] == right['input_sha256'] == data['environment']['input_sha256'],
                  original_environment_unchanged=existing_version == importlib.metadata.version('faster-coco-eval'))
    if not all(checks.values()):
        raise RuntimeError(f'Parity isolation failed: {checks}')
    identical = not (metric_diffs or config_diffs or setup_diffs)
    parity = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  command='& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/scorer_parity.py',
                  commands=commands, runs=runs, checks=checks,
                  metric_values_compared=sum(1 + len(r['ap_by_class']) for r in left['cases'].values()),
                  cases_compared=len(left['cases']), comparison='Exact equality, no tolerance or rounding',
                  metric_differences=metric_diffs, configuration_differences=config_diffs,
                  fixture_differences=setup_diffs, differences_from_recorded_suite=baseline_diffs,
                  every_numeric_result_identical=identical,
                  strategic_conclusions_changed=False if identical else 'Requires difference review')
    data['scorer_version_parity'] = parity
    data_path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    sys.path.insert(0, str(CHALLENGE))
    from evaluator_probes.report import write_report
    write_report(data, DIRECTORY / 'PROBE_RESULTS.md')
    print(json.dumps({k: parity[k] for k in ('cases_compared', 'metric_values_compared',
                     'every_numeric_result_identical', 'metric_differences', 'configuration_differences',
                     'fixture_differences', 'differences_from_recorded_suite', 'strategic_conclusions_changed')}, indent=2))


if __name__ == '__main__':
    main()
