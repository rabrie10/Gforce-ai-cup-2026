"""Independent scalar reconstruction from source JSON, not experiment math helpers.

The final metamorphic checks deliberately import the production feature API; all
saved prediction/IoU/summary reconstruction above those checks is independent.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from statistics import median, mean
from collections import defaultdict

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
LIMITS = [3840, 2160, 3840, 2160]


def read(name):
    with (OUT / (name+'.csv')).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-9), (a, b)


def rect(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1])


def independent_iou(a, b):
    # Center/half-width intersections; primary uses corner/vector arithmetic.
    ax, ay, aw, ah = rect(a)
    bx, by, bw, bh = rect(b)
    if min(aw, ah) <= 0:
        return 0.
    iw = max(0, min(ax+aw/2, bx+bw/2)-max(ax-aw/2, bx-bw/2))
    ih = max(0, min(ay+ah/2, by+bh/2)-max(ay-ah/2, by-bh/2))
    return iw*ih/(aw*ah+bw*bh-iw*ih)


def main():
    result = json.loads((OUT / 'motion_oracle_results.json').read_text())
    for path, expected in result['input_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    boxes = {}
    frames = []
    for path in sorted((ROOT / 'drone-flyby/src/helsinki/annotations').glob('*.json')):
        d = json.loads(path.read_text())
        frames.append(d['frame'])
        for a in d['annotations']:
            key = a['object_id'], d['frame']
            assert key not in boxes
            b = a['bbox']
            assert b[0] < b[2] and b[1] < b[3]
            assert all(0 <= x <= lim for x, lim in zip(b, LIMITS))
            boxes[key] = b
    assert frames == list(range(25)) and len(boxes) == 259
    classes = sorted({c for c, _ in boxes})
    meta = json.loads((ROOT / 'drone-flyby/src/helsinki/run_metadata.json').read_text())
    assert meta['total_objects'] == len(classes) == 16
    assert meta['object_totals'] == {c: 1 for c in classes}
    for c in classes:
        ff = sorted(t for cl, t in boxes if cl == c)
        assert ff == list(range(min(ff), max(ff)+1))

    donor_rows = read('donor_updates')
    assert len(donor_rows) == 17*24
    updates = {r['update_id']: r for r in donor_rows}
    assert len(updates) == len(donor_rows)
    estimates = {}
    for r in donor_rows:
        c, k = r['excluded_class'], int(r['end_frame'])
        assert int(r['start_frame']) == k-1
        expected = sorted(cl for cl in classes if cl != c and (cl, k-1) in boxes and (cl, k) in boxes)
        assert expected == r['donors'].split('|')
        assert c not in expected and int(r['donor_count']) == len(expected)
        assert int(r['sufficient']) == int(len(expected) >= 1)
        velocities = [(rect(boxes[cl, k])[0]-rect(boxes[cl, k-1])[0],
                       rect(boxes[cl, k])[1]-rect(boxes[cl, k-1])[1]) for cl in expected]
        g = tuple(median(v[i] for v in velocities) for i in [0, 1])
        estimates[c, k] = g
        for i, key in enumerate(['dx', 'dy']):
            close(r[key], g[i])
            close(r['mad_'+key], median(abs(v[i]-g[i]) for v in velocities))
        close(r['residual_median'], median(math.dist(v, g) for v in velocities))

    rows = read('prediction_results')
    expected_keys = {(c, t, u-t, m) for c, t in boxes for u in range(t+1, 25)
                     if (c, u) in boxes for m in range(5)}
    actual_keys = set()
    cohorts = defaultdict(list)
    expected_common = set()
    for r in rows:
        c, t, u, h = r['target_class'], int(r['origin_frame']), int(r['target_frame']), int(r['horizon'])
        m = int(r['model'][1])
        key = c, t, h, m
        assert key not in actual_keys
        actual_keys.add(key)
        assert u == t+h and h > 0 and (c, u) in boxes
        close(r['seconds'], h/3)
        history_ok = (c, t-1) in boxes
        available = not ((m in [1, 4] and not history_ok) or (m == 2 and t == 0))
        assert int(r['available']) == available
        feature_frames = [int(x) for x in r['target_feature_frames'].split('|')]
        assert feature_frames == ([t-1, t] if m in [1, 4] and history_ok else [t])
        assert max(feature_frames) <= t
        # Model 4 returns early for missing target history, before donor access.
        kk = [] if m == 4 and not history_ok else ([t] if m in [2, 4] else []) + (list(range(t+1, u+1)) if m in [3, 4] else [])
        assert r['update_ids'] == '|'.join(f'{c}:{k}' for k in kk)
        if kk:
            assert int(r['donor_count_min']) == min(int(updates[f'{c}:{k}']['donor_count']) if k else 0 for k in kk)
        else:
            assert r['donor_count_min'] == ''
        for k in kk:
            if k:
                assert c not in updates[f'{c}:{k}']['donors'].split('|')
        cohorts[c, t, h].append(r)
        if history_ok:
            expected_common.add((c, t, h))
        assert int(r['common_cohort']) == history_ok
        if not available:
            assert not r['iou'] and not r['raw_x1'] and not r['valid']
            assert r['unavailable_reason'] == ('missing_target_history' if m in [1, 4] else 'insufficient_donors')
            continue
        b = boxes[c, t]
        cx, cy, w, ht = rect(b)
        if m in [1, 4]:
            px, py, pw, ph = rect(boxes[c, t-1])
            vx, vy, vw, vh = cx-px, cy-py, w-pw, ht-ph
        if m == 1:
            cx, cy, w, ht = cx+h*vx, cy+h*vy, w+h*vw, ht+h*vh
        if m == 2:
            gx, gy = estimates[c, t]
            cx, cy = cx+h*gx, cy+h*gy
        if m in [3, 4]:
            for k in range(t+1, u+1):
                gx, gy = estimates[c, k]
                cx, cy = cx+gx, cy+gy
        if m == 4:
            gx, gy = estimates[c, t]
            cx, cy, w, ht = cx+h*(vx-gx), cy+h*(vy-gy), w+h*vw, ht+h*vh
        raw = [cx-w/2, cy-ht/2, cx+w/2, cy+ht/2]
        pred = [max(0, min(lim, x)) for lim, x in zip(LIMITS, raw)]
        for i, coord in enumerate(['x1', 'y1', 'x2', 'y2']):
            close(r['raw_'+coord], raw[i])
            close(r['pred_'+coord], pred[i])
        truth = boxes[c, u]  # accessed after independent feature reconstruction
        score = independent_iou(pred, truth)
        assert 0 <= score <= 1
        close(r['iou'], score)
        assert int(r['valid']) == (score >= .5)
        assert int(r['prediction_legal']) == (pred[2] > pred[0] and pred[3] > pred[1])
        tx, ty, tw, th = rect(truth)
        close(r['center_error'], math.hypot(cx-tx, cy-ty))
        close(r['center_error_short_side'], math.hypot(cx-tx, cy-ty)/min(tw, th))
        close(r['width_error'], w-tw)
        close(r['height_error'], ht-th)
    assert actual_keys == expected_keys and len(rows) == result['prediction_rows'] == 12280
    assert len(expected_common) == 2213
    assert all(len(v) == 5 for v in cohorts.values())

    group_specs = [('horizon_summary', ['model', 'horizon'], False),
                   ('common_cohort_summary', ['model', 'horizon'], True),
                   ('per_class_summary', ['model', 'horizon', 'target_class'], True),
                   ('size_summary', ['model', 'horizon', 'origin_size_bin'], True),
                   ('target_size_summary', ['model', 'horizon', 'target_size_bin'], True),
                   ('boundary_summary', ['model', 'horizon', 'any_boundary'], True)]
    summary_count = 0
    for filename, keys, common in group_specs:
        groups = defaultdict(list)
        for r in rows:
            if not common or int(r['common_cohort']):
                groups[tuple(r[k] for k in keys)].append(r)
        saved = read(filename)
        assert len(saved) == len(groups)
        for s in saved:
            group = groups[tuple(s[k] for k in keys)]
            a = [r for r in group if int(r['available'])]
            assert int(s['eligible_count']) == len(group) and int(s['available_count']) == len(a)
            success = sum(int(r['valid']) for r in a)
            assert int(s['valid_count']) == success
            close(s['availability_fraction'], len(a)/len(group))
            if a:
                close(s['valid_fraction'], success/len(a))
                close(s['mean_iou'], mean(float(r['iou']) for r in a))
                close(s['median_iou'], median(float(r['iou']) for r in a))
                close(s['median_center_error'], median(float(r['center_error']) for r in a))
            summary_count += 1
    expected_counts = {1:(227,227), 2:(207,212), 3:(179,197), 4:(146,182), 5:(104,168), 6:(58,155), 9:(0,117)}
    for h, pair in expected_counts.items():
        rr = [r for r in rows if int(r['model'][1]) == 1 and int(r['horizon']) == h and int(r['available'])]
        assert (sum(int(r['valid']) for r in rr), len(rr)) == pair
    audit = {(r['class'], r['origin_frame'], r['target_frame']): float(r['iou'])
             for r in csv.DictReader((ROOT / 'drone-flyby/scene_audit/stale_box_iou.csv').open()) if r['method'] == 'velocity'}
    for r in rows:
        if int(r['model'][1]) == 1 and int(r['available']):
            close(r['iou'], audit[r['target_class'], r['origin_frame'], r['target_frame']])
    loo = read('loo_spatial_generalization')
    assert len(loo) == 243
    assert {(r['target_class'], int(r['end_frame'])) for r in loo} == {(c, k) for c, k in boxes if (c, k-1) in boxes}
    for r in loo:
        c, k = r['target_class'], int(r['end_frame'])
        assert int(r['start_frame']) == k-1
        gx, gy = estimates[c, k]
        b = boxes[c, k-1]
        pred = [max(0, min(lim, v+d)) for lim, v, d in zip(LIMITS, b, [gx, gy, gx, gy])]
        close(r['iou'], independent_iou(pred, boxes[c, k]))
        p, q = rect(b), rect(boxes[c, k])
        close(r['center_error'], math.hypot(p[0]+gx-q[0], p[1]+gy-q[1]))

    # Behavioral leakage check: mutate every future target box per origin while
    # retaining other-object GT. Re-estimation must leave held-out features invariant.
    import run_experiment as primary
    b = primary.load_data()
    base_updates = primary.shared_updates(b)
    poisoning_origins = 0
    for c, t in sorted(b):
        if (c, t+1) not in b:
            continue
        changed = {key: (value + primary.np.array([101, 73, 119, 87]) if key[0] == c and key[1] > t else value.copy()) for key, value in b.items()}
        changed_updates = primary.shared_updates(changed)
        for k in range(1, 25):
            assert base_updates[f'{c}:{k}'] == changed_updates[f'{c}:{k}']
        history = {k: v for (cl, k), v in changed.items() if cl == c and k <= t}
        donor = {k: changed_updates[f'{c}:{k}'] for k in range(1, 25)}
        for m in range(5):
            h = max(k for cl, k in b if cl == c)-t
            pred, _, _ = primary.predict(m, t, h, history, donor)
            saved = next(r for r in cohorts[c, t, h] if int(r['model'][1]) == m)
            if pred is not None:
                for coord, x in zip(['x1', 'y1', 'x2', 'y2'], pred):
                    close(saved['raw_'+coord], x)
        poisoning_origins += 1
    # Explicit no-support and fixed-shared equivalence checks.
    history = {0: primary.np.array([10., 20., 30., 50.]), 1: primary.np.array([12., 25., 34., 58.])}
    for m in [2, 3, 4]:
        pred, _, reason = primary.predict(m, 1, 2, history, {})
        assert pred is None and reason == 'insufficient_donors'
    fixed = {k: dict(sufficient=1, dx=3., dy=4.) for k in range(1, 5)}
    for h in [1, 2, 3]:
        cv, _, _ = primary.predict(1, 1, h, history, fixed)
        hybrid, _, _ = primary.predict(4, 1, h, history, fixed)
        assert primary.np.array_equal(cv, hybrid)
    figures = list((OUT / 'figures').glob('*.png'))
    assert len(figures) == 8 and all(p.stat().st_size > 10000 for p in figures)
    report = dict(status='PASS', independent_prediction_rows=len(rows), common_cohort_keys=len(expected_common),
                  independently_checked_summary_rows=summary_count, donor_updates=len(updates),
                  independently_checked_loo_transitions=len(loo), audit_cv_rows_matched=len(audit),
                  future_target_poisoning_origins=poisoning_origins, no_support_checks='PASS',
                  fixed_shared_hybrid_equals_cv='PASS', unchanged_inputs=len(result['input_sha256']), figures=len(figures))
    (OUT / 'validation_results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    validate_derived()


def validate_derived():
    """Supplementary checks can run without repeating the passed prediction gate."""
    rows = read('prediction_results')
    by_key = defaultdict(dict)
    for r in rows:
        by_key[r['target_class'], r['origin_frame'], r['horizon']][r['model']] = r
    pairs = read('pairwise_common_cohorts')
    for r in pairs:
        a, b = r['model_a'], r['model_b']
        group = [g for key, g in by_key.items() if key[2] == r['horizon'] and int(g[a]['available']) and int(g[b]['available'])]
        assert len(group) == int(r['common_count'])
        assert sum(int(g[a]['valid']) for g in group) == int(r['a_valid'])
        assert sum(int(g[b]['valid']) for g in group) == int(r['b_valid'])
        assert sum(int(g[a]['valid']) and not int(g[b]['valid']) for g in group) == int(r['a_only_valid'])
        assert sum(int(g[b]['valid']) and not int(g[a]['valid']) for g in group) == int(r['b_only_valid'])
    boxes = {}
    for path in (ROOT / 'drone-flyby/src/helsinki/annotations').glob('*.json'):
        doc = json.loads(path.read_text())
        for ann in doc['annotations']:
            boxes[ann['object_id'], doc['frame']] = ann['bbox']
    dec = read('motion_decomposition')
    updates = {r['update_id']: r for r in read('donor_updates')}
    assert len(dec) == 243
    for r in dec:
        c, k = r['target_class'], int(r['end_frame'])
        a, b = rect(boxes[c, k-1]), rect(boxes[c, k])
        g = updates[f'ALL:{k}']
        for i, key in enumerate(['dx', 'dy']):
            close(r[key], b[i]-a[i])
            close(r['residual_'+key], b[i]-a[i]-float(g[key]))
        close(r['raw_magnitude'], math.dist(a[:2], b[:2]))
        close(r['residual_magnitude'], math.hypot(float(r['residual_dx']), float(r['residual_dy'])))
        close(r['width_ratio'], b[2]/a[2])
        close(r['height_ratio'], b[3]/a[3])
    result = json.loads((OUT / 'motion_oracle_results.json').read_text())
    raw = [float(r['raw_magnitude']) for r in dec]
    residual = [float(r['residual_magnitude']) for r in dec]
    close(result['residual_to_raw_median_ratio'], median(residual)/median(raw))
    close(result['uncentered_motion_energy_reduction'], 1-sum(x*x for x in residual)/sum(x*x for x in raw))
    diag = json.loads((OUT / 'additional_diagnostics.json').read_text())
    used = {i for r in rows if int(r['available']) for i in r['update_ids'].split('|') if i}
    assert len(used) == diag['used_donor_updates']
    for key, stats in diag['used_donor_statistics'].items():
        values = [float(updates[i][key]) for i in used]
        close(stats['median'], median(values))
        close(stats['min'], min(values))
        close(stats['max'], max(values))
        close(stats['mean'], mean(values))
    loo = read('loo_spatial_generalization')
    for name, stats in diag['spatial_correlations'].items():
        subset = loo if name == 'all' else [r for r in loo if not int(r['boundary'])]
        assert stats['n'] == len(subset)
        for axis, error in [('x', 'dx'), ('y', 'dy')]:
            xx = [float(r[axis]) for r in subset]
            yy = [float(r['error_'+error]) for r in subset]
            mx, my = mean(xx), mean(yy)
            coefficient = sum((x-mx)*(y-my) for x, y in zip(xx, yy))/math.sqrt(sum((x-mx)**2 for x in xx)*sum((y-my)**2 for y in yy))
            close(stats[f'corr_{axis}_signed_{error}_error'], coefficient)
    checked = dict(status='PASS', pairwise_common_summary_rows=len(pairs),
                   decomposition_rows=len(dec), donor_dispersion_and_spatial_correlations='PASS')
    (OUT / 'derived_validation_results.json').write_text(json.dumps(checked, indent=2)+'\n')
    print(json.dumps(checked, indent=2))


if __name__ == '__main__':
    validate_derived() if '--derived-only' in sys.argv else main()
