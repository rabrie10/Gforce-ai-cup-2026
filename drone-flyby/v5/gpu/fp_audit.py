import json
RES = "/srv/gforce/v5-perception-results/hosted-epoch3"
manual = json.load(open("/srv/gforce/v5-perception-results/manual_targets.json"))["targets"]
by_cap = {}
for t in manual:
    by_cap.setdefault(t["capture"], []).append(t)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.


out = {"frames": {}, "method": "Emitted boxes classified as background-FP when they do not overlap (IoU<0.1) any manually-verified target. Scores read from saved pipeline diagnostics. combined=discovery*target_prob; final conf=combined*posterior_max (age 0)."}
gate_levels = [0.5, 0.7, 0.9, 0.95]
for cap in (60, 125):
    diag = json.load(open(f"{RES}/hosted_{cap:05d}_candidates.json"))
    raw = diag["candidates"]
    targets = [t["box"] for t in by_cap[cap]]
    emitted = [c for c in raw if c.get("emitted")]
    rows = []
    for c in emitted:
        lb = c["local_box"]
        best_t = max([iou(lb, tb) for tb in targets], default=0.0)
        is_fp = best_t < 0.10
        cvmax = max(c["class_scores"]) if c["class_scores"] else 0.0
        combined = c["discovery_score"] * (c["target_probability"] or 0)
        rows.append({"raw_rank": c["raw_rank"], "cand_rank": c["candidate_rank"],
                     "top1": c["top3"][0]["class"], "top1_score": round(c["top3"][0]["score"], 3),
                     "discovery": round(c["discovery_score"], 3), "target_prob": round(c["target_probability"], 3),
                     "posterior_max": round(cvmax, 3), "combined_disc*tgt": round(combined, 3),
                     "final_conf": round(combined*cvmax, 3),
                     "overlaps_real_target_iou": round(best_t, 3), "is_background_FP": is_fp})
    n_fp = sum(1 for r in rows if r["is_background_FP"])
    tp_vals = [r["target_prob"] for r in rows]
    # threshold sensitivity: how many emitted survive each target gate; and (diagnostic) how many manual targets' ORACLE tp would survive
    sens = []
    for g in gate_levels:
        survive = sum(1 for r in rows if r["target_prob"] >= g)
        sens.append({"target_gate": g, "emitted_boxes_surviving": survive,
                     "background_FPs_surviving": sum(1 for r in rows if r["target_prob"] >= g and r["is_background_FP"])})
    out["frames"][cap] = {"emitted_total": len(rows), "background_FPs": n_fp,
                          "real_targets_correctly_emitted(IoU>=0.1)": len(rows)-n_fp,
                          "emitted_target_prob_min": round(min(tp_vals), 3) if tp_vals else None,
                          "emitted_target_prob_max": round(max(tp_vals), 3) if tp_vals else None,
                          "emitted_final_conf_max": round(max((r["final_conf"] for r in rows), default=0), 3),
                          "gate_sensitivity": sens, "emitted_boxes": rows}
json.dump(out, open("/srv/gforce/v5-perception-results/audit/background_fp_audit.json", "w"), indent=2)
for cap in (60, 125):
    f = out["frames"][cap]
    print(f"--- capture {cap:05d}: emitted={f['emitted_total']} background_FP={f['background_FPs']} correctly_on_target={f['real_targets_correctly_emitted(IoU>=0.1)']} tp_range=[{f['emitted_target_prob_min']},{f['emitted_target_prob_max']}] max_final_conf={f['emitted_final_conf_max']}")
    print("    gate sensitivity (emitted surviving / of which FP):", ", ".join(f"g{s['target_gate']}:{s['emitted_boxes_surviving']}/{s['background_FPs_surviving']}" for s in f["gate_sensitivity"]))
print("WROTE /srv/gforce/v5-perception-results/audit/background_fp_audit.json")
