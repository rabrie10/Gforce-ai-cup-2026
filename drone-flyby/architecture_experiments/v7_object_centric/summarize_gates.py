import json, collections, sys
rows = json.load(open("/workspace/audit-l1l2/gate_ablation_v7_e15.json"))
base = json.load(open("/workspace/audit-l1l2/gate_ablation_ms1.json"))
print("V7 @ det_conf .15 -- best candidate per frozen target, and what the classifier says:")
print("%-20s %9s %8s %8s  %s" % ("target", "det_score", "min_side", "target_p", "class_top1"))
seen = {}
for r in rows:
    if r["det_conf"] != 0.15 or r["best_iou"] < 0.5 or not r["best_target"]:
        continue
    k = r["best_target"]
    if k not in seen or r["best_iou"] > seen[k]["best_iou"]:
        seen[k] = r
for k in sorted(seen):
    r = seen[k]
    flag = "" if (r["min_side"] >= 16 and r["target_p"] >= 0.5) else "   <-- still gated out"
    print("%-20s %9.3f %8.1f %8.2f  %s%s" % (k, r["score"], r["min_side"], r["target_p"], r["top1"], flag))


def fps(rs, mp=16):
    return collections.Counter(r["top1"] for r in rs
                               if r["det_conf"] == 0.15 and r["best_iou"] < 0.2 and r["min_side"] >= mp and r["target_p"] >= 0.5)


print("\nV7 FP candidates (no overlap with any frozen target), by emitted class:", dict(fps(rows).most_common(8)))
print("ms1 FP candidates, by emitted class:", dict(fps(base).most_common(8)))
print("\nV7 FP total over 20 frames:", sum(fps(rows).values()), " ms1:", sum(fps(base).values()))
