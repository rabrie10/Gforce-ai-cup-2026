"""Helsinki GT-only motion diagnostic. No image access, inference or evaluator import."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT / '.deps'))
import numpy as np

ROOT = OUT.parents[2]
DATA = ROOT / 'drone-flyby/src/helsinki'
AUDIT = ROOT / 'drone-flyby/scene_audit'
LIMITS = np.array([3840., 2160., 3840., 2160.])
MODELS = ['M0_HOLD', 'M1_INDEPENDENT_CV', 'M2_SHARED_OPEN_LOOP',
          'M3_ORACLE_REFRESHED_SHARED', 'M4_ORACLE_REFRESHED_SHARED_RESIDUAL']
EXPECTED = {1: (227, 227), 2: (207, 212), 3: (179, 197), 4: (146, 182),
            5: (104, 168), 6: (58, 155), 9: (0, 117)}


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def distribution(x):
    x = np.asarray(x, dtype=float)
    return dict(n=len(x), min=float(x.min()), p10=float(np.percentile(x, 10)),
                median=float(np.median(x)), mean=float(x.mean()),
                p90=float(np.percentile(x, 90)), max=float(x.max()))


def center(b):
    return (b[:2] + b[2:]) / 2


def size(b):
    return b[2:] - b[:2]


def corners(c, s):
    return np.r_[c - s / 2, c + s / 2]


def overlap(a, b):
    if np.any(size(a) <= 0):
        return 0.
    inter = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    return float(inter / (np.prod(size(a)) + np.prod(size(b)) - inter))


def boundary(b):
    return int(min(b[0], b[1], 3840-b[2], 2160-b[3]) <= 1)


def size_bin(b):
    s = min(size(b)) / 4  # audit's L0 apparent short side
    return '<4' if s < 4 else '4-<8' if s < 8 else '8-<16' if s < 16 else '16-32' if s <= 32 else '>32'


def load_data():
    boxes = {}
    for path in sorted((DATA / 'annotations').glob('*.json')):
        doc = json.loads(path.read_text())
        for ann in doc['annotations']:
            key = (ann['object_id'], doc['frame'])
            assert key not in boxes
            b = np.array(ann['bbox'], dtype=float)
            assert np.all(size(b) > 0) and np.all(b >= 0) and np.all(b <= LIMITS)
            boxes[key] = b
    return boxes


def baseline_gate(boxes):
    """Runs before any new model. Also compares every CV row with the saved audit."""
    audit = {(r['class'], int(r['origin_frame']), int(r['target_frame'])): float(r['iou'])
             for r in csv.DictReader((AUDIT / 'stale_box_iou.csv').open()) if r['method'] == 'velocity'}
    rows = []
    for (c, t), b in sorted(boxes.items()):
        if (c, t-1) not in boxes:
            continue
        for u in range(t+1, 25):
            if (c, u) in boxes:
                value = overlap(np.clip(b+(u-t)*(b-boxes[c, t-1]), 0, LIMITS), boxes[c, u])
                assert abs(value-audit[c, t, u]) < 1e-12, (c, t, u)
                rows.append((u-t, value))
    result = {}
    for h, expected in EXPECTED.items():
        values = [v for k, v in rows if k == h]
        measured = (sum(v >= .5 for v in values), len(values))
        assert measured == expected, (h, measured, expected)
        result[str(h)] = dict(valid=measured[0], eligible=measured[1])
    assert len(rows) == len(audit)
    result['all_audit_cv_rows_matched'] = len(rows)
    save_json('baseline_reproduction.json', result)
    return result


def shared_updates(boxes):
    """ALL is descriptive only; predictions can access held-out updates only."""
    updates = {}
    classes = sorted({c for c, _ in boxes})
    for excluded in ['ALL'] + classes:
        for k in range(1, 25):
            donors = [c for c in classes if c != excluded and (c, k-1) in boxes and (c, k) in boxes]
            vel = np.array([center(boxes[c, k])-center(boxes[c, k-1]) for c in donors])
            ratios = np.array([size(boxes[c, k])/size(boxes[c, k-1]) for c in donors])
            nonedge = [c for c in donors if not (boundary(boxes[c, k-1]) or boundary(boxes[c, k]))]
            clean = np.array([size(boxes[c, k])/size(boxes[c, k-1]) for c in nonedge])
            g = np.median(vel, axis=0) if donors else None
            mad = np.median(abs(vel-g), axis=0) if donors else None
            key = f'{excluded}:{k}'
            updates[key] = dict(update_id=key, excluded_class=excluded, start_frame=k-1, end_frame=k,
                donors='|'.join(donors), donor_count=len(donors), sufficient=int(len(donors) >= 1),
                dx=float(g[0]) if donors else None, dy=float(g[1]) if donors else None,
                mad_dx=float(mad[0]) if donors else None, mad_dy=float(mad[1]) if donors else None,
                residual_median=float(np.median(np.linalg.norm(vel-g, axis=1))) if donors else None,
                magnitude=float(np.linalg.norm(g)) if donors else None,
                direction_deg=float(np.degrees(np.arctan2(g[1], g[0]))) if donors else None,
                width_ratio=float(np.median(ratios[:, 0])) if donors else None,
                height_ratio=float(np.median(ratios[:, 1])) if donors else None,
                clean_scale_count=len(clean),
                clean_width_ratio=float(np.median(clean[:, 0])) if len(clean) else None,
                clean_height_ratio=float(np.median(clean[:, 1])) if len(clean) else None,
                clean_width_ratio_mad=float(np.median(abs(clean[:, 0]-np.median(clean[:, 0])))) if len(clean) else None,
                clean_height_ratio_mad=float(np.median(abs(clean[:, 1]-np.median(clean[:, 1])))) if len(clean) else None)
    return updates


def predict(model, t, h, history, donor_updates):
    """Feature-only API: target history ends at t, updates exclude the target.

    Future target box is unavailable to this function. Propagate latent centers;
    clipping occurs only for the requested output, never recursively on state.
    """
    assert max(history) <= t
    b = history[t]
    ids = []
    if model in (1, 4) and t-1 not in history:
        return None, ids, 'missing_target_history'
    if model in (2, 4):
        ids.append(t)
    if model in (3, 4):
        ids.extend(range(t+1, t+h+1))
    if any(k not in donor_updates or not donor_updates[k]['sufficient'] for k in ids):
        return None, ids, 'insufficient_donors'
    shared = lambda k: np.array([donor_updates[k]['dx'], donor_updates[k]['dy']])
    c, s = center(b), size(b)
    if model == 1:
        return b+h*(b-history[t-1]), ids, ''
    if model == 2:
        c = c+h*shared(t)
    if model in (3, 4):
        c = c+sum((shared(k) for k in range(t+1, t+h+1)), start=np.zeros(2))
    if model == 4:
        c = c+h*(center(b)-center(history[t-1])-shared(t))
        s = s+h*(size(b)-size(history[t-1]))
    return corners(c, s), ids, ''


def evaluate(boxes, updates):
    rows = []
    for (c, t), b in sorted(boxes.items()):
        history = {k: v for (cl, k), v in boxes.items() if cl == c and k <= t}
        donor = {k: updates[f'{c}:{k}'] for k in range(1, 25)}
        for u in range(t+1, 25):
            if (c, u) not in boxes:
                continue
            for m, name in enumerate(MODELS):
                raw, ids, reason = predict(m, t, u-t, history, donor)
                # Only now retrieve future target GT, solely for evaluation.
                truth = boxes[c, u]
                available = raw is not None
                pred = np.clip(raw, 0, LIMITS) if available else None
                ce = float(np.linalg.norm(center(raw)-center(truth))) if available else None
                r = dict(target_class=c, origin_frame=t, target_frame=u, horizon=u-t, seconds=(u-t)/3,
                    model=name, available=int(available), unavailable_reason=reason,
                    target_feature_frames='|'.join(map(str, [t-1, t] if m in (1, 4) and t-1 in history else [t])),
                    update_ids='|'.join(f'{c}:{k}' for k in ids),
                    donor_count_min=min((donor[k]['donor_count'] if k in donor else 0 for k in ids), default=None),
                    donor_residual_median_mean=float(np.mean([donor[k]['residual_median'] for k in ids])) if ids and available else None,
                    origin_size_bin=size_bin(b), target_size_bin=size_bin(truth),
                    origin_boundary=boundary(b), target_boundary=boundary(truth),
                    any_boundary=int(any(boundary(boxes[c, k]) for k in [t-1, t, u] if (c, k) in boxes)),
                    origin_cx=float(center(b)[0]), origin_cy=float(center(b)[1]),
                    iou=overlap(pred, truth) if available else None,
                    valid=int(overlap(pred, truth) >= .5) if available else None,
                    prediction_legal=int(np.all(size(pred) > 0)) if available else None,
                    center_error=ce, center_error_short_side=ce/min(size(truth)) if available else None,
                    clipped_center_error=float(np.linalg.norm(center(pred)-center(truth))) if available else None,
                    width_error=float(size(raw)[0]-size(truth)[0]) if available else None,
                    height_error=float(size(raw)[1]-size(truth)[1]) if available else None)
                for i, coord in enumerate(['x1', 'y1', 'x2', 'y2']):
                    r['raw_'+coord] = float(raw[i]) if available else None
                    r['pred_'+coord] = float(pred[i]) if available else None
                rows.append(r)
    cohorts = defaultdict(list)
    for r in rows:
        cohorts[r['target_class'], r['origin_frame'], r['horizon']].append(r)
    for group in cohorts.values():
        common = int(all(r['available'] for r in group))
        for r in group:
            r['common_cohort'] = common
    return rows


def summary(rows, groupkeys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in groupkeys)].append(r)
    result = []
    for key, group in sorted(groups.items()):
        a = [r for r in group if r['available']]
        result.append(dict(zip(groupkeys, key), eligible_count=len(group), available_count=len(a),
            availability_fraction=len(a)/len(group), valid_count=sum(r['valid'] for r in a),
            valid_fraction=sum(r['valid'] for r in a)/len(a) if a else None,
            mean_iou=float(np.mean([r['iou'] for r in a])) if a else None,
            median_iou=float(np.median([r['iou'] for r in a])) if a else None,
            median_center_error=float(np.median([r['center_error'] for r in a])) if a else None,
            median_normalized_center_error=float(np.median([r['center_error_short_side'] for r in a])) if a else None,
            mean_abs_width_error=float(np.mean([abs(r['width_error']) for r in a])) if a else None,
            mean_abs_height_error=float(np.mean([abs(r['height_error']) for r in a])) if a else None))
    return result


def decomposition(boxes, updates):
    rows, loo = [], []
    for (c, k), b in sorted(boxes.items()):
        if (c, k-1) not in boxes:
            continue
        prev = boxes[c, k-1]
        v = center(b)-center(prev)
        g = np.array([updates[f'ALL:{k}'][z] for z in ['dx', 'dy']])
        held = updates[f'{c}:{k}']
        hv = np.array([held[z] for z in ['dx', 'dy']])
        residual = v-g
        rows.append(dict(target_class=c, start_frame=k-1, end_frame=k,
            x=float(center(prev)[0]), y=float(center(prev)[1]), dx=float(v[0]), dy=float(v[1]),
            shared_dx=float(g[0]), shared_dy=float(g[1]), residual_dx=float(residual[0]), residual_dy=float(residual[1]),
            raw_magnitude=float(np.linalg.norm(v)), residual_magnitude=float(np.linalg.norm(residual)),
            width_ratio=float(size(b)[0]/size(prev)[0]), height_ratio=float(size(b)[1]/size(prev)[1]),
            boundary=int(boundary(b) or boundary(prev))))
        predicted = np.clip(corners(center(prev)+hv, size(prev)), 0, LIMITS)
        loo.append(dict(target_class=c, start_frame=k-1, end_frame=k, update_id=held['update_id'],
            donor_count=held['donor_count'], x=float(center(prev)[0]), y=float(center(prev)[1]),
            error_dx=float(hv[0]-v[0]), error_dy=float(hv[1]-v[1]),
            center_error=float(np.linalg.norm(hv-v)), iou=overlap(predicted, b),
            valid=int(overlap(predicted, b) >= .5), boundary=int(boundary(b) or boundary(prev))))
    return rows, loo


def plots(common, dec, shared, loo, classes):
    os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.grid': True, 'grid.alpha': .2})
    figdir = OUT / 'figures'
    figdir.mkdir(exist_ok=True)
    labels = ['M0 Hold', 'M1 Independent CV', 'M2 Shared open loop',
              'M3 ORACLE REFRESHED SHARED MOTION', 'M4 ORACLE REFRESHED SHARED MOTION + residual']
    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(figdir / (name+'.png'))
        plt.close(fig)
    for metric, title, ylabel, fname in [
        ('valid_fraction', 'IoU >= 0.50 fraction (not first-failure survival)', 'Fraction', 'iou_success'),
        ('median_iou', 'Median IoU', 'IoU', 'median_iou'),
        ('median_center_error', 'Median latent-center error', 'Source pixels', 'center_error')]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, label in zip(MODELS, labels):
            rr = sorted([r for r in common if r['model'] == name], key=lambda r: r['horizon'])
            ax.plot([r['horizon'] for r in rr], [r[metric] for r in rr], '.-', label=label)
        ax.set(title=title+' | Helsinki common cohorts', xlabel='Horizon (frames; 3 FPS)', ylabel=ylabel)
        if metric != 'median_center_error':
            ax.set_ylim(0, 1.03)
        ax.legend(fontsize=8)
        finish(fig, fname)
    fig, ax = plt.subplots(figsize=(9, 5))
    for values, label in [([r['raw_magnitude'] for r in dec], 'Raw velocity'),
                          ([r['residual_magnitude'] for r in dec], 'All-object descriptive residual'),
                          ([r['center_error'] for r in loo], 'Held-out residual')]:
        ax.ecdf(values, label=label)
    ax.set(xlabel='Source pixels / frame', ylabel='Empirical cumulative fraction', title='Motion decomposition | 243 transitions')
    ax.legend()
    finish(fig, 'velocity_distribution')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric in zip(axes.flat, ['dx', 'dy', 'magnitude', 'direction_deg']):
        ax.plot([r['end_frame']/3 for r in shared], [r[metric] for r in shared], '.-')
        ax.set(xlabel='Transition end (seconds)', ylabel=metric, title='All-object descriptive shared '+metric)
    finish(fig, 'shared_motion_time')
    fig, ax = plt.subplots(figsize=(11, 7))
    names = sorted({r['target_class'] for r in classes if r['horizon'] == 6})
    matrix = [[next(r['valid_fraction'] for r in classes if r['target_class'] == c and r['horizon'] == 6 and r['model'] == m) for m in MODELS] for c in names]
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(5), ['M0', 'M1', 'M2', 'M3 oracle', 'M4 oracle'])
    for y, row in enumerate(matrix):
        for x, v in enumerate(row):
            ax.text(x, y, f'{v:.0%}', ha='center', va='center', color='white' if v < .5 else 'black')
    ax.set_title('h=6 common-cohort success by class (classes with eligible samples)')
    fig.colorbar(im, ax=ax, label='IoU >= 0.50 fraction')
    finish(fig, 'per_class_h6')
    fig, ax = plt.subplots(figsize=(11, 6))
    q = ax.quiver([r['x'] for r in loo], [r['y'] for r in loo], [r['error_dx'] for r in loo],
                  [r['error_dy'] for r in loo], [r['center_error'] for r in loo], angles='xy', scale_units='xy', scale=.08, cmap='plasma')
    ax.set(xlim=(0, 3840), ylim=(2160, 0), xlabel='Source x', ylabel='Source y', title='Held-out translation error vectors (arrows magnified 12.5x)')
    fig.colorbar(q, ax=ax, label='Error px/frame')
    finish(fig, 'spatial_residuals')
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in ['clean_width_ratio', 'clean_height_ratio']:
        ax.plot([r['end_frame'] for r in shared], [r[metric] for r in shared], '.-', label=metric)
    ax.axhline(1, color='black', linewidth=.7)
    ax.set(xlabel='Transition end frame', ylabel='Median size ratio', title='Non-boundary donors: size trend (no scale model fitted)')
    ax.legend()
    finish(fig, 'scale_trend')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    sources = sorted((DATA / 'annotations').glob('*.json')) + [DATA / 'run_metadata.json']
    sources += [AUDIT / name for name in ['scene_audit.json', 'DATASET_SCENE_AUDIT.md', 'MEASUREMENTS.md', 'stale_box_iou.csv']]
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    boxes = load_data()
    baseline = baseline_gate(boxes)
    print('Baseline reproduction PASS:', baseline)
    if args.baseline_only:
        return
    updates = shared_updates(boxes)
    rows = evaluate(boxes, updates)
    common_rows = [r for r in rows if r['common_cohort']]
    horizons = summary(rows, ['model', 'horizon'])
    common = summary(common_rows, ['model', 'horizon'])
    classes = summary(common_rows, ['model', 'horizon', 'target_class'])
    dec, loo = decomposition(boxes, updates)
    shared = [r for r in updates.values() if r['excluded_class'] == 'ALL']
    for name, table in [('prediction_results', rows), ('horizon_summary', horizons),
                        ('common_cohort_summary', common), ('per_class_summary', classes),
                        ('size_summary', summary(common_rows, ['model', 'horizon', 'origin_size_bin'])),
                        ('target_size_summary', summary(common_rows, ['model', 'horizon', 'target_size_bin'])),
                        ('boundary_summary', summary(common_rows, ['model', 'horizon', 'any_boundary'])),
                        ('donor_updates', list(updates.values())), ('motion_decomposition', dec),
                        ('shared_motion_by_frame', shared), ('loo_spatial_generalization', loo)]:
        save_csv(name+'.csv', table)
    raw = np.array([[r['dx'], r['dy']] for r in dec])
    residual = np.array([[r['residual_dx'], r['residual_dy']] for r in dec])
    dr = distribution([r['raw_magnitude'] for r in dec])
    ds = distribution([r['residual_magnitude'] for r in dec])
    spatial = []
    for axis, cut in [('x', 1920), ('y', 1080)]:
        for side in [0, 1]:
            subset = [r for r in loo if int(r[axis] >= cut) == side]
            spatial.append(dict(axis=axis, side='high' if side else 'low', n=len(subset),
                valid=sum(r['valid'] for r in subset), center_error=distribution([r['center_error'] for r in subset])))
    clean = [r for r in dec if not r['boundary']]
    result = dict(scope='Helsinki only; GT-only; future other-object GT is ORACLE REFRESHED SHARED MOTION',
        python=platform.python_version(), numpy=np.__version__, input_sha256=hashes,
        baseline=baseline, annotation_count=len(boxes), prediction_rows=len(rows),
        common_prediction_rows=len(common_rows), transition_count=len(dec),
        raw_velocity=dr, residual_velocity=ds,
        shared_magnitude=distribution([r['magnitude'] for r in shared]),
        residual_to_raw_median_ratio=ds['median']/dr['median'],
        uncentered_motion_energy_reduction=1-float(np.sum(residual**2)/np.sum(raw**2)),
        centered_variance_reduction=1-float(np.var(residual, axis=0).sum()/np.var(raw, axis=0).sum()),
        loo=dict(n=len(loo), valid=sum(r['valid'] for r in loo), median_iou=float(np.median([r['iou'] for r in loo])),
                 center_error=distribution([r['center_error'] for r in loo])),
        donor_count=distribution([r['donor_count'] for r in updates.values() if r['excluded_class'] != 'ALL']),
        shared_time={k: distribution([r[k] for r in shared]) for k in ['dx', 'dy', 'magnitude', 'direction_deg']},
        scale_nonboundary={k: distribution([r[k] for r in clean]) for k in ['width_ratio', 'height_ratio']},
        spatial=spatial, common_cohort_summary=common,
        definitions=dict(valid='IoU >= 0.50; fraction denominator is available_count',
            eligible='all GT origin/future pairs, including history/donor-unavailable cases',
            common='all five models available at same class, origin, horizon',
            center_error='unclipped latent center versus GT center; clipped error also saved',
            boundary='<=1 pixel from any edge at target history or evaluation frame',
            minimum_donors=1))
    save_json('motion_oracle_results.json', result)
    plots(common, dec, shared, loo, classes)
    assert hashes == {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in hashes}
    print('Saved', len(rows), 'prediction rows;', len(common_rows), 'common rows;', len(dec), 'transitions')


if __name__ == '__main__':
    main()
