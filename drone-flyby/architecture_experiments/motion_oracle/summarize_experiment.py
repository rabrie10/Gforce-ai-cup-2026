"""Additional descriptive tables from saved results; does not refit predictions."""
from pathlib import Path
import csv
import itertools
import json
import math
from statistics import median, mean
from collections import defaultdict

OUT = Path(__file__).resolve().parent


def read(name):
    return list(csv.DictReader((OUT / (name+'.csv')).open(newline='', encoding='utf-8')))


def save(name, rows):
    with (OUT / (name+'.csv')).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def spread(values):
    return dict(n=len(values), min=min(values), median=median(values), mean=mean(values), max=max(values))


def corr(x, y):
    a, b = mean(x), mean(y)
    return sum((u-a)*(v-b) for u, v in zip(x, y))/math.sqrt(sum((u-a)**2 for u in x)*sum((v-b)**2 for v in y))


def main():
    rows = read('prediction_results')
    models = sorted({r['model'] for r in rows})
    by_key = defaultdict(dict)
    for r in rows:
        by_key[r['target_class'], r['origin_frame'], r['horizon']][r['model']] = r
    pairs = []
    for a, b in itertools.combinations(models, 2):
        for h in range(1, 25):
            rr = [g for key, g in by_key.items() if int(key[2]) == h and int(g[a]['available']) and int(g[b]['available'])]
            if not rr:
                continue
            pairs.append(dict(model_a=a, model_b=b, horizon=h, common_count=len(rr),
                a_valid=sum(int(g[a]['valid']) for g in rr), b_valid=sum(int(g[b]['valid']) for g in rr),
                a_only_valid=sum(int(g[a]['valid']) and not int(g[b]['valid']) for g in rr),
                b_only_valid=sum(int(g[b]['valid']) and not int(g[a]['valid']) for g in rr)))
    save('pairwise_common_cohorts', pairs)
    loo = read('loo_spatial_generalization')
    used_ids = {i for r in rows if int(r['available']) for i in r['update_ids'].split('|') if i}
    donor = [r for r in read('donor_updates') if r['update_id'] in used_ids]
    shared = read('shared_motion_by_frame')
    dec = read('motion_decomposition')
    scale_classes = []
    for c in sorted({r['target_class'] for r in dec}):
        rr = [r for r in dec if r['target_class'] == c and not int(r['boundary'])]
        if rr:
            scale_classes.append(dict(target_class=c, nonboundary_pairs=len(rr),
                median_width_ratio=median(float(r['width_ratio']) for r in rr),
                median_height_ratio=median(float(r['height_ratio']) for r in rr)))
    save('scale_by_class', scale_classes)
    spatial = {}
    for subset_name, rr in [('all', loo), ('nonboundary', [r for r in loo if not int(r['boundary'])])]:
        spatial[subset_name] = dict(n=len(rr), valid=sum(int(r['valid']) for r in rr),
            corr_x_signed_dx_error=corr([float(r['x']) for r in rr], [float(r['error_dx']) for r in rr]),
            corr_y_signed_dy_error=corr([float(r['y']) for r in rr], [float(r['error_dy']) for r in rr]))
    diagnostics = dict(used_donor_updates=len(donor),
        used_donor_statistics={k: spread([float(r[k]) for r in donor]) for k in ['donor_count', 'mad_dx', 'mad_dy', 'residual_median']},
        spatial_correlations=spatial,
        clean_shared_scale={k: spread([float(r[k]) for r in shared]) for k in ['clean_width_ratio', 'clean_height_ratio', 'clean_width_ratio_mad', 'clean_height_ratio_mad']},
        shared_first_last={k:[float(shared[0][k]),float(shared[-1][k])] for k in ['dx','dy','magnitude','direction_deg']},
        note='Pearson correlations are descriptive and confounded by class, time and boundary; no significance claim.')
    (OUT / 'additional_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
