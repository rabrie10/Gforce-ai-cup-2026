"""Dependency-free offline scoring; never imported by the serving pipeline."""
from collections import defaultdict
from statistics import mean


def iou(a, b):
    area = lambda x: max(0, x[2] - x[0]) * max(0, x[3] - x[1])
    intersection = area((max(a[0], b[0]), max(a[1], b[1]),
                         min(a[2], b[2]), min(a[3], b[3])))
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.0


def contains(region, box):
    return (region[0] <= box[0] < box[2] <= region[2]
            and region[1] <= box[1] < box[3] <= region[3])


def lift(box, region):
    sx, sy = (region[2] - region[0]) / 960, (region[3] - region[1]) / 540
    return (region[0] + box[0] * sx, region[1] + box[1] * sy,
            region[0] + box[2] * sx, region[1] + box[3] * sy)


def size_bucket(box):
    side = min(box[2] - box[0], box[3] - box[1])
    return '<32' if side < 32 else '32-64' if side < 64 else '>=64'


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for bucket in ('all', row['size_source_short_side']):
            groups[(row['mode'], row['level'], row['budget'], bucket)].append(row)
    return [dict(mode=k[0], level=k[1], budget=k[2], size=k[3], n=len(v),
                 recall50=mean(r['best_iou'] >= .5 for r in v),
                 mean_best_iou=mean(r['best_iou'] for r in v))
            for k, v in sorted(groups.items())]


def gate(rows):
    """Predeclared engineering gate on paired oracle object appearances.

    Zoom must help K8 recall AND localization versus matched-scale and L0.
    These are exploratory repeated observations, not independent samples.
    """
    by_key = {(r['frame'], r['object_index'], r['mode'], r['level'], r['budget']): r
              for r in rows}
    results = []
    for level in (1, 2):
        pairs = []
        for key, native in by_key.items():
            frame, obj, mode, lev, budget = key
            if (mode, lev, budget) != ('native', level, 8):
                continue
            matched = by_key.get((frame, obj, 'matched', level, 8))
            l0 = by_key.get((frame, obj, 'matched', 0, 8))
            if matched and l0:
                pairs.append((native['best_iou'], matched['best_iou'], l0['best_iou']))
        n = len(pairs)
        recall = [mean(p[i] >= .5 for p in pairs) if n else 0 for i in range(3)]
        loc = [mean(p[i] for p in pairs) if n else 0 for i in range(3)]
        passed = (n >= 30 and recall[0] - recall[1] >= .10 - 1e-9
                  and recall[0] - recall[2] >= .10 - 1e-9
                  and loc[0] - loc[1] >= .05 - 1e-9
                  and loc[0] - loc[2] >= .05 - 1e-9)
        results.append(dict(level=level, paired_n=n, recall_native_matched_l0=recall,
                            iou_native_matched_l0=loc, passed=passed))
    return dict(decision='GO' if any(r['passed'] for r in results) else 'STOP',
                levels=results, scope='Permission to investigate Phase B, not a score claim')
