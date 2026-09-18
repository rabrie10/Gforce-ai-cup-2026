"""GT-only held-out similarity. No image access, evaluator, or prior-package writes."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / 'motion_oracle'
ROOT = OUT.parents[2]
DATA = ROOT / 'drone-flyby/src/helsinki'
sys.path[:0] = [str(OUT / '.deps'), str(PRIOR / '.deps')]
import numpy as np

LIMITS = np.array([3840., 2160., 3840., 2160.])
MODELS = ['CV', 'T_FIXED', 'T_RES_CV', 'S_FIXED', 'S_RES_CV', 'S_SCALE', 'S_RES_SCALE']
CONFIG = dict(min_donors=3, min_rms_spread=100., min_minor_spread=25.,
              min_eigen_ratio=.001, huber_delta_px=3., reweight_iterations=20)
LABELS = dict(CV='Independent CV', T_FIXED='Translation / fixed size',
    T_RES_CV='Translation + residual / CV size', S_FIXED='Similarity / fixed size',
    S_RES_CV='Similarity + residual / CV size', S_SCALE='Similarity / scale size',
    S_RES_SCALE='Similarity + residual / scale size')


def save_json(name, doc):
    (OUT / name).write_text(json.dumps(doc, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def save_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def center(b):
    return (b[:2]+b[2:])/2


def size(b):
    return b[2:]-b[:2]


def corners(c, s):
    return np.r_[c-s/2, c+s/2]


def iou(a, b):
    if np.any(size(a) <= 0):
        return 0.
    inter = np.prod(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])))
    return float(inter/(np.prod(size(a))+np.prod(size(b))-inter))


def boundary(b):
    return int(min(b[0], b[1], 3840-b[2], 2160-b[3]) <= 1)


def size_bin(b):
    s = min(size(b))/4
    return '<4' if s < 4 else '4-<8' if s < 8 else '8-<16' if s < 16 else '16-32' if s <= 32 else '>32'


def stats(x):
    x = np.array(x, dtype=float)
    if not len(x):
        return dict(n=0)
    return dict(n=len(x), min=float(x.min()), median=float(np.median(x)), mean=float(x.mean()),
                p90=float(np.percentile(x, 90)), max=float(x.max()))


def load_data():
    boxes = {}
    for path in sorted((DATA / 'annotations').glob('*.json')):
        d = json.loads(path.read_text())
        for a in d['annotations']:
            key = a['object_id'], d['frame']
            assert key not in boxes
            b = np.array(a['bbox'], dtype=float)
            assert np.all(size(b) > 0) and np.all(b >= 0) and np.all(b <= LIMITS)
            boxes[key] = b
    return boxes


def spatial_support(x, weights):
    """Spread is sqrt eigenvalues of population weighted center covariance."""
    if len(x) == 0:
        return dict(rms_spread=0., minor_spread=0., major_spread=0., eigen_ratio=0.)
    z = x-np.average(x, axis=0, weights=weights)
    cov = (z.T*weights) @ z / sum(weights)
    eig = np.maximum(0., np.linalg.eigvalsh(cov))
    return dict(rms_spread=float(np.sqrt(sum(eig))), minor_spread=float(np.sqrt(eig[0])),
                major_spread=float(np.sqrt(eig[1])), eigen_ratio=float(eig[0]/eig[1]) if eig[1] else 0.)


def sufficient(s):
    return (s['rms_spread'] >= CONFIG['min_rms_spread'] and
            s['minor_spread'] >= CONFIG['min_minor_spread'] and
            s['eigen_ratio'] >= CONFIG['min_eigen_ratio'])


def fit_similarity(x, y):
    """Orientation-preserving similarity y=[[a,-b],[b,a]]x+d.

    Fixed Huber IRLS in source pixels. All candidates remain donors; downweighting
    uses other-object fit errors only. No target-based fitting/tuning.
    """
    x, y = np.asarray(x, dtype=float).reshape(-1, 2), np.asarray(y, dtype=float).reshape(-1, 2)
    weights = np.ones(len(x))
    spread = spatial_support(x, weights)
    base = dict(sufficient=0, reason='', **spread, weighted_rms_spread=None,
        weighted_minor_spread=None, weighted_eigen_ratio=None, effective_donors=None,
        a=None, b=None, tx=None, ty=None, scale=None, rotation_deg=None,
        fit_median=None, fit_rms=None, fit_p90=None, weights='', donor_errors='')
    if len(x) < CONFIG['min_donors']:
        return dict(base, reason='too_few_donors')
    if not sufficient(spread):
        return dict(base, reason='degenerate_spatial_support')
    def solve(w):
        xm, ym = np.average(x, axis=0, weights=w), np.average(y, axis=0, weights=w)
        xc, yc = x-xm, y-ym
        design = np.empty((2*len(x), 2))
        design[::2] = np.c_[xc[:, 0], -xc[:, 1]]
        design[1::2] = np.c_[xc[:, 1], xc[:, 0]]
        sw = np.repeat(np.sqrt(w), 2)
        a, b = np.linalg.lstsq(design*sw[:, None], yc.ravel()*sw, rcond=None)[0]
        mat = np.array([[a, -b], [b, a]])
        return float(a), float(b), ym-mat @ xm, np.linalg.norm(x @ mat.T+(ym-mat @ xm)-y, axis=1)
    for _ in range(CONFIG['reweight_iterations']):
        a, b, shift, errors = solve(weights)
        weights = np.minimum(1., CONFIG['huber_delta_px']/np.maximum(errors, 1e-15))
    a, b, shift, errors = solve(weights)
    weighted = spatial_support(x, weights)
    effective = float(sum(weights)**2/sum(weights**2))
    out = dict(base, weighted_rms_spread=weighted['rms_spread'],
        weighted_minor_spread=weighted['minor_spread'], weighted_eigen_ratio=weighted['eigen_ratio'],
        effective_donors=effective, weights='|'.join(format(v, '.17g') for v in weights),
        donor_errors='|'.join(format(v, '.17g') for v in errors))
    if effective < CONFIG['min_donors'] or not sufficient(weighted):
        return dict(out, reason='degenerate_weighted_support')
    scale = float(np.hypot(a, b))
    if not np.isfinite(scale) or scale <= 1e-12:
        return dict(out, reason='collapsed_transform')
    return dict(out, sufficient=1, a=a, b=b, tx=float(shift[0]), ty=float(shift[1]),
        scale=scale, rotation_deg=float(np.degrees(np.arctan2(b, a))),
        fit_median=float(np.median(errors)), fit_rms=float(np.sqrt(np.mean(errors**2))),
        fit_p90=float(np.percentile(errors, 90)))


def build_updates(boxes, target):
    """Only donors enter fit arrays; target coordinates are never accessed."""
    classes = sorted({c for c, _ in boxes if c != target})
    updates = {}
    for k in range(1, 25):
        donors = [c for c in classes if (c, k-1) in boxes and (c, k) in boxes]
        x = np.array([center(boxes[c, k-1]) for c in donors]).reshape(-1, 2)
        y = np.array([center(boxes[c, k]) for c in donors]).reshape(-1, 2)
        velocity = y-x
        g = np.median(velocity, axis=0) if donors else None
        fit = fit_similarity(x, y)
        updates[k] = dict(update_id=f'{target}:{k}', excluded_class=target, start_frame=k-1, end_frame=k,
            donors='|'.join(donors), donor_count=len(donors),
            boundary_donor_count=sum(boundary(boxes[c, k-1]) or boundary(boxes[c, k]) for c in donors),
            translation_sufficient=int(bool(donors)),
            dx=float(g[0]) if donors else None, dy=float(g[1]) if donors else None,
            mad_dx=float(np.median(abs(velocity[:, 0]-g[0]))) if donors else None,
            mad_dy=float(np.median(abs(velocity[:, 1]-g[1]))) if donors else None,
            translation_residual_median=float(np.median(np.linalg.norm(velocity-g, axis=1))) if donors else None,
            **fit)
    return updates


def apply_transform(u, p):
    return np.array([u['a']*p[0]-u['b']*p[1]+u['tx'], u['b']*p[0]+u['a']*p[1]+u['ty']])


def predict(model, t, h, history, updates):
    """Feature-only API. Frozen residuals use fixed image-axis pixels/step."""
    assert max(history) <= t
    residual_model = '_RES_' in model
    if (model == 'CV' or residual_model) and t-1 not in history:
        return None, [], 'missing_target_history'
    ids = [] if model == 'CV' else ([t] if residual_model else [])+list(range(t+1, t+h+1))
    support = 'sufficient' if model.startswith('S_') else 'translation_sufficient'
    if any(k not in updates or not updates[k][support] for k in ids):
        return None, ids, 'insufficient_or_degenerate_donors'
    b = history[t]
    if model == 'CV':
        return b+h*(b-history[t-1]), ids, ''
    c, s = center(b), size(b)
    g = lambda k: np.array([updates[k]['dx'], updates[k]['dy']])
    if model.startswith('T_'):
        c = c+sum((g(k) for k in range(t+1, t+h+1)), start=np.zeros(2))
        if residual_model:
            c = c+h*(center(b)-center(history[t-1])-g(t))
    else:
        residual = center(b)-apply_transform(updates[t], center(history[t-1])) if residual_model else np.zeros(2)
        for k in range(t+1, t+h+1):
            c = apply_transform(updates[k], c)+residual
            if model.endswith('_SCALE'):
                s = s*updates[k]['scale']
    if model.endswith('_CV'):
        s = size(b)+h*(size(b)-size(history[t-1]))
    return corners(c, s), ids, ''


def preservation_manifest():
    paths = subprocess.check_output(['git', 'ls-files', '--', str(PRIOR.relative_to(ROOT)).replace('\\', '/')], cwd=ROOT, text=True).splitlines()
    paths += [str(p.relative_to(ROOT)).replace('\\', '/') for p in sorted((DATA / 'annotations').glob('*.json'))]
    paths += [str((DATA / 'run_metadata.json').relative_to(ROOT)).replace('\\', '/')]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def control_gate(boxes, updates):
    mapping = {'M1_INDEPENDENT_CV': 'CV', 'M3_ORACLE_REFRESHED_SHARED': 'T_FIXED',
               'M4_ORACLE_REFRESHED_SHARED_RESIDUAL': 'T_RES_CV'}
    counts = defaultdict(int)
    max_error = 0.
    for old in read_csv(PRIOR / 'prediction_results.csv'):
        if old['model'] not in mapping:
            continue
        c, t, h = old['target_class'], int(old['origin_frame']), int(old['horizon'])
        history = {k: b for (cl, k), b in boxes.items() if cl == c and k <= t}
        raw, _, _ = predict(mapping[old['model']], t, h, history, updates[c])
        assert (raw is not None) == bool(int(old['available']))
        if raw is not None:
            error = max(abs(float(old['raw_'+coord])-raw[i]) for i, coord in enumerate(['x1','y1','x2','y2']))
            max_error = max(max_error, float(error))
            assert error < 1e-9
            value = iou(np.clip(raw, 0, LIMITS), boxes[c, t+h])
            assert abs(value-float(old['iou'])) < 1e-12
        counts[mapping[old['model']]] += 1
    expected = json.loads((PRIOR / 'baseline_reproduction.json').read_text())
    result = dict(status='PASS', compared_rows=dict(counts), max_raw_coordinate_difference=max_error, prior_cv_gate=expected)
    save_json('control_reproduction.json', result)
    return result


def evaluate(boxes, updates):
    rows = []
    for (c, t), b in sorted(boxes.items()):
        history = {k: v for (cl, k), v in boxes.items() if cl == c and k <= t}
        for u in range(t+1, 25):
            if (c, u) not in boxes:
                continue
            group = []
            for model in MODELS:
                raw, ids, reason = predict(model, t, u-t, history, updates[c])
                truth = boxes[c, u]  # evaluation-only access after prediction
                available = raw is not None
                pred = np.clip(raw, 0, LIMITS) if available else None
                ce = float(np.linalg.norm(center(raw)-center(truth))) if available else None
                history_frames = [t-1, t] if (model == 'CV' or '_RES_' in model) and t-1 in history else [t]
                r = dict(target_class=c, origin_frame=t, target_frame=u, horizon=u-t, seconds=(u-t)/3,
                    model=model, oracle_refreshed=int(model != 'CV'), available=int(available), unavailable_reason=reason,
                    target_feature_frames='|'.join(map(str, history_frames)), update_ids='|'.join(f'{c}:{k}' for k in ids),
                    donor_count_min=min((updates[c][k]['donor_count'] for k in ids if k in updates[c]), default=None),
                    origin_size_bin=size_bin(b), target_size_bin=size_bin(truth),
                    any_boundary=int(any(boundary(boxes[c, k]) for k in [t-1, t, u] if (c, k) in boxes)),
                    origin_boundary=boundary(b), target_boundary=boundary(truth),
                    origin_cx=float(center(b)[0]), origin_cy=float(center(b)[1]),
                    iou=iou(pred, truth) if available else None, valid=int(iou(pred, truth) >= .5) if available else None,
                    prediction_legal=int(np.all(size(pred) > 0)) if available else None,
                    center_error=ce, center_error_short_side=ce/min(size(truth)) if available else None,
                    clipped_center_error=float(np.linalg.norm(center(pred)-center(truth))) if available else None,
                    width_error=float(size(raw)[0]-size(truth)[0]) if available else None,
                    height_error=float(size(raw)[1]-size(truth)[1]) if available else None)
                for i, coord in enumerate(['x1','y1','x2','y2']):
                    r['raw_'+coord] = float(raw[i]) if available else None
                    r['pred_'+coord] = float(pred[i]) if available else None
                group.append(r)
            for r in group:
                r['common_cohort'] = int(all(v['available'] for v in group))
            rows.extend(group)
    return rows


def summarize(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, group in sorted(groups.items()):
        a = [r for r in group if r['available']]
        out.append(dict(zip(keys, key), eligible_count=len(group), available_count=len(a),
            availability_fraction=len(a)/len(group), valid_count=sum(r['valid'] for r in a),
            valid_fraction=sum(r['valid'] for r in a)/len(a) if a else None,
            mean_iou=float(np.mean([r['iou'] for r in a])) if a else None,
            median_iou=float(np.median([r['iou'] for r in a])) if a else None,
            mean_center_error=float(np.mean([r['center_error'] for r in a])) if a else None,
            median_center_error=float(np.median([r['center_error'] for r in a])) if a else None,
            median_normalized_center_error=float(np.median([r['center_error_short_side'] for r in a])) if a else None,
            mean_abs_width_error=float(np.mean([abs(r['width_error']) for r in a])) if a else None,
            mean_abs_height_error=float(np.mean([abs(r['height_error']) for r in a])) if a else None))
    return out


def loo_diagnostics(boxes, updates):
    rows = []
    for (c, k), b in sorted(boxes.items()):
        if (c, k-1) not in boxes:
            continue
        prev = boxes[c, k-1]
        for model in ['T_FIXED', 'S_FIXED', 'S_SCALE']:
            raw, _, _ = predict(model, k-1, 1, {k-1: prev}, updates[c])
            available = raw is not None
            error = center(raw)-center(b) if available else None
            rows.append(dict(target_class=c, start_frame=k-1, end_frame=k, model=model,
                update_id=f'{c}:{k}', available=int(available), x=float(center(prev)[0]), y=float(center(prev)[1]),
                boundary=int(boundary(prev) or boundary(b)), center_error=float(np.linalg.norm(error)) if available else None,
                error_dx=float(error[0]) if available else None, error_dy=float(error[1]) if available else None,
                iou=iou(np.clip(raw, 0, LIMITS), b) if available else None,
                valid=int(iou(np.clip(raw, 0, LIMITS), b) >= .5) if available else None))
    common_keys = { (r['target_class'], r['end_frame']) for r in rows if r['available'] == 0 }
    summaries = []
    for model in ['T_FIXED', 'S_FIXED', 'S_SCALE']:
        for cohort in ['all', 'nonboundary']:
            rr = [r for r in rows if r['model'] == model and (cohort == 'all' or not r['boundary'])
                  and (r['target_class'], r['end_frame']) not in common_keys]
            d = dict(model=model, cohort=cohort, count=len(rr), valid_count=sum(r['valid'] for r in rr),
                     median_iou=float(np.median([r['iou'] for r in rr])), center_error=stats([r['center_error'] for r in rr]))
            for axis, component in [('x','dx'), ('y','dy')]:
                xx, yy = np.array([r[axis] for r in rr]), np.array([r['error_'+component] for r in rr])
                d['corr_'+axis+'_error_'+component] = float(np.corrcoef(xx, yy)[0, 1])
                by_frame = defaultdict(list)
                for i, r in enumerate(rr):
                    by_frame[r['end_frame']].append(i)
                xd, yd = xx.copy(), yy.copy()
                for ids in by_frame.values():
                    xd[ids] -= np.mean(xx[ids]); yd[ids] -= np.mean(yy[ids])
                d['within_transition_corr_'+axis+'_error_'+component] = float(np.corrcoef(xd, yd)[0, 1])
            summaries.append(d)
    return rows, summaries


def pairwise(rows):
    keys = defaultdict(dict)
    for r in rows:
        keys[r['target_class'], r['origin_frame'], r['horizon']][r['model']] = r
    out = []
    for a, b in itertools.combinations(MODELS, 2):
        for h in range(1, 25):
            rr = [g for key, g in keys.items() if key[2] == h and g[a]['available'] and g[b]['available']]
            if rr:
                out.append(dict(model_a=a, model_b=b, horizon=h, count=len(rr),
                    a_valid=sum(g[a]['valid'] for g in rr), b_valid=sum(g[b]['valid'] for g in rr),
                    a_only=sum(g[a]['valid'] and not g[b]['valid'] for g in rr),
                    b_only=sum(g[b]['valid'] and not g[a]['valid'] for g in rr)))
    return out


def plots(common, loo, updates, classes):
    os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':130, 'font.size':10, 'axes.grid':True, 'grid.alpha':.2})
    (OUT / 'figures').mkdir(exist_ok=True)
    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / (name+'.png'))
        plt.close(fig)
    main_models = MODELS[:5]
    for metric, fname, ylabel in [('valid_fraction','iou_success','IoU >=0.50 fraction'),
                                  ('median_iou','median_iou','Median IoU'),
                                  ('median_center_error','center_error','Median center error (source px)')]:
        fig, axes = plt.subplots(1,2,figsize=(13,5))
        for ax, maximum in zip(axes,[6,23]):
            for m in main_models:
                rr = sorted([r for r in common if r['model']==m and r['horizon']<=maximum],key=lambda r:r['horizon'])
                ax.plot([r['horizon'] for r in rr],[r[metric] for r in rr],'.-',label=LABELS[m])
            ax.set(xlabel='Horizon (frames; 3 FPS)',ylabel=ylabel)
            if metric!='median_center_error': ax.set_ylim(0,1.03)
        axes[0].legend(fontsize=8)
        fig.suptitle('Matched size policies | all shared models use ORACLE REFRESHED SHARED MOTION')
        finish(fig,fname)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax, names in zip(axes,[['S_FIXED','S_SCALE'],['S_RES_CV','S_RES_SCALE']]):
        for m in names:
            rr=sorted([r for r in common if r['model']==m],key=lambda r:r['horizon'])
            ax.plot([r['horizon'] for r in rr],[r['valid_fraction'] for r in rr],'.-',label=LABELS[m])
        ax.set(xlabel='Horizon (frames)',ylabel='IoU >=0.50 fraction',ylim=(0,1.03))
        ax.legend(fontsize=9)
    fig.suptitle('ORACLE REFRESHED SHARED MOTION | size ablation: identical centers within each panel')
    finish(fig,'scale_size_ablation')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,m in zip(axes,['T_FIXED','S_FIXED']):
        rr=[r for r in loo if r['model']==m and r['available']]
        q=ax.quiver([r['x'] for r in rr],[r['y'] for r in rr],[r['error_dx'] for r in rr],
            [r['error_dy'] for r in rr],[r['center_error'] for r in rr],angles='xy',scale_units='xy',scale=.08,cmap='plasma',clim=(0,45))
        ax.set(xlim=(0,3840),ylim=(2160,0),xlabel='Source x',ylabel='Source y',title=LABELS[m])
    fig.suptitle('Held-out errors; arrows magnified 12.5x | ORACLE REFRESHED SHARED MOTION')
    # Reserve a separate colorbar axis so it never overlaps source-space vectors.
    fig.subplots_adjust(top=.80,right=.86,wspace=.25)
    fig.colorbar(q,cax=fig.add_axes([.90,.16,.015,.60]),label='Error px')
    fig.savefig(OUT/'figures/spatial_residuals.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,axis,comp in zip(axes,['x','y'],['dx','dy']):
        for m in ['T_FIXED','S_FIXED']:
            rr=[r for r in loo if r['model']==m and r['available'] and not r['boundary']]
            ax.scatter([r[axis] for r in rr],[r['error_'+comp] for r in rr],s=12,alpha=.65,label=LABELS[m])
        ax.set(xlabel='Source '+axis,ylabel='Signed '+comp+' error (px)'); ax.legend(fontsize=8)
    fig.suptitle('Non-boundary held-out spatial structure | ORACLE REFRESHED SHARED MOTION')
    finish(fig,'spatial_correlations')
    fig,ax=plt.subplots(figsize=(9,5))
    for m in ['T_FIXED','S_FIXED']:
        ax.ecdf([r['center_error'] for r in loo if r['model']==m and r['available']],label=LABELS[m])
    ax.set(xlabel='Held-out center error px',ylabel='Empirical cumulative fraction',title='ORACLE REFRESHED SHARED MOTION: one-step transfer')
    ax.legend(); finish(fig,'loo_error_distribution')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,key in zip(axes,['scale','rotation_deg']):
        for k in range(1,25):
            values=[r[key] for r in updates if r['end_frame']==k and r['sufficient']]
            ax.plot([k]*len(values),values,'.',alpha=.35,color='tab:blue')
        ax.set(xlabel='Transition end frame',ylabel=key)
    fig.suptitle('Held-out similarity transforms (overlapping donor fits, not independent samples)')
    finish(fig,'transform_time')
    fig,ax=plt.subplots(figsize=(11,7))
    names=sorted({r['target_class'] for r in classes if r['horizon']==6})
    matrix=[[next(r['valid_fraction'] for r in classes if r['target_class']==c and r['horizon']==6 and r['model']==m) for m in MODELS] for c in names]
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_yticks(range(len(names)),names); ax.set_xticks(range(len(MODELS)),MODELS,rotation=25,ha='right')
    for y,row in enumerate(matrix):
        for x,v in enumerate(row): ax.text(x,y,f'{v:.0%}',ha='center',va='center',color='white' if v<.5 else 'black')
    ax.set_title('h=6 common-cohort success | every T/S model is oracle refreshed')
    fig.colorbar(im,ax=ax); finish(fig,'per_class_h6')


def main():
    before = preservation_manifest()
    save_json('preserved_inputs.json', before)
    boxes = load_data()
    classes = sorted({c for c, _ in boxes})
    updates = {c: build_updates(boxes, c) for c in classes}
    gate = control_gate(boxes, updates)
    print('Controls PASS',gate['compared_rows'],flush=True)
    rows = evaluate(boxes, updates)
    common_rows = [r for r in rows if r['common_cohort']]
    common = summarize(common_rows, ['model','horizon'])
    class_summary = summarize(common_rows, ['model','horizon','target_class'])
    loo, spatial = loo_diagnostics(boxes, updates)
    flat = [r for uu in updates.values() for r in uu.values()]
    for name, table in [('prediction_results',rows),('donor_transforms',flat),
        ('horizon_summary',summarize(rows,['model','horizon'])),('common_cohort_summary',common),
        ('per_class_summary',class_summary),('size_summary',summarize(common_rows,['model','horizon','origin_size_bin'])),
        ('boundary_summary',summarize(common_rows,['model','horizon','any_boundary'])),
        ('target_size_summary',summarize(common_rows,['model','horizon','target_size_bin'])),
        ('loo_spatial_generalization',loo),('pairwise_common_cohorts',pairwise(rows))]:
        save_csv(name+'.csv',table)
    used = {i for r in rows if r['available'] for i in r['update_ids'].split('|') if i}
    used_fits = [r for r in flat if r['update_id'] in used]
    result = dict(scope='Helsinki only; all T/S models are ORACLE REFRESHED SHARED MOTION',
        config=CONFIG, models=LABELS, python=platform.python_version(), numpy=np.__version__,
        residual_convention='r=c_t - F_t(c_(t-1)); add unchanged image-axis pixels after each future transform',
        scale_size_policy='multiply origin width/height by product of future similarity scales; no rotation-induced AABB enlargement',
        prediction_rows=len(rows), common_prediction_rows=len(common_rows),
        available_by_model={m:sum(r['available'] for r in rows if r['model']==m) for m in MODELS},
        controls=gate, spatial=spatial, common_cohort_summary=common,
        transform_count=len(flat), unavailable_transforms=sum(not r['sufficient'] for r in flat),
        used_transform_count=len(used_fits),
        used_transform_statistics={k:stats([r[k] for r in used_fits if r[k] is not None]) for k in
            ['donor_count','rms_spread','minor_spread','eigen_ratio','effective_donors','fit_median','fit_rms','scale','rotation_deg']})
    save_json('similarity_oracle_results.json', result)
    plots(common,loo,flat,class_summary)
    assert before == preservation_manifest(), 'Prior artifacts or data changed'
    print('Saved',len(rows),'rows;',len(common_rows),'common rows; prior artifacts unchanged',flush=True)


if __name__ == '__main__':
    if '--plots-only' in sys.argv:
        def numeric_rows(name):
            rr=read_csv(OUT/(name+'.csv'))
            for r in rr:
                for k,v in r.items():
                    try: r[k]=float(v)
                    except ValueError: pass
            return rr
        plots(json.loads((OUT/'similarity_oracle_results.json').read_text())['common_cohort_summary'],
              numeric_rows('loo_spatial_generalization'),numeric_rows('donor_transforms'),numeric_rows('per_class_summary'))
    else:
        main()
