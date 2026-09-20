import json, os, cv2, numpy as np
from v5.pipeline import Config
from v5.visual_expert import VisualExpert
from dtos import OBJECT_CLASSES

H = "/hosted-v3"
RES = "/results/hosted-epoch3"
OUT = "/results/audit"; os.makedirs(OUT, exist_ok=True)
CLOSE = os.path.join(OUT, "closeups"); os.makedirs(CLOSE, exist_ok=True)
files = {60: "00060_80324c03bcc65ff1d04c510e_180324c03bcc65ff1d04c510e_60_0_1920_1080.png",
         125: "00125_80324c03bcc65ff1d04c510e_80324c03bcc65ff1d04c510e_125_0_1920_1080.png"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.


def center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)
def inside(pt, b): return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


cfg = Config(assets="/assets", temporal=False, active_camera=False)
expert = VisualExpert(cfg.assets, device="cpu", head_name=cfg.head_name, threads=2)

manual = json.load(open("/results/manual_targets.json"))["targets"]
by_cap = {}
for t in manual:
    by_cap.setdefault(t["capture"], []).append(t)

report = []
for cap, fn in files.items():
    img = cv2.imread(os.path.join(H, "images", fn))
    diag = json.load(open(f"{RES}/hosted_{cap:05d}_candidates.json"))
    raw = diag["candidates"]
    meta = json.load(open(os.path.join(H, "frames", fn.replace(".png", ".json"))))
    v3 = []
    for p in meta.get("emitted", []):
        box = p.get("box", p.get("bbox"))
        if box is None:
            continue
        scale = (960, 540, 960, 540) if max(box) <= 1 else (.25, .25, .25, .25)
        v3.append({"box": [b*s for b, s in zip(box, scale)],
                   "label": str(p.get("class", p.get("object_id", p.get("top", "?"))))})
    final = [{"box": [b*s for b, s in zip(a["bbox"], (960, 540, 960, 540))],
              "label": a["object_id"], "conf": a["confidence"]}
             for a in diag["response"]["annotations"]]
    for t in by_cap[cap]:
        mb = t["box"]; mc = center(mb)
        ious = sorted([(iou(mb, c["local_box"]), c) for c in raw], key=lambda z: z[0], reverse=True)
        best_iou, best = ious[0]
        n010 = sum(1 for i, _ in ious if i >= 0.10)
        n030 = sum(1 for i, _ in ious if i >= 0.30)
        n050 = sum(1 for i, _ in ious if i >= 0.50)
        contain = sorted([c for c in raw if inside(mc, c["local_box"]) or inside(center(c["local_box"]), mb)],
                         key=lambda c: c["discovery_score"], reverse=True)
        A = {"best_raw_iou": round(best_iou, 3), "best_candidate_raw_rank": best["raw_rank"],
             "best_candidate_local_box": [round(v, 1) for v in best["local_box"]],
             "best_candidate_discovery_score": round(best["discovery_score"], 4),
             "num_raw_iou>=0.10": n010, "num_raw_iou>=0.30": n030, "num_raw_iou>=0.50": n050,
             "num_center_containment_matches": len(contain),
             "best_containment_candidate": ({"raw_rank": contain[0]["raw_rank"], "candidate_rank": contain[0]["candidate_rank"],
                 "selected": contain[0]["selected"], "discovery_score": round(contain[0]["discovery_score"], 4),
                 "iou": round(iou(mb, contain[0]["local_box"]), 3)} if contain else None)}
        matched = best if best_iou >= 0.10 else (contain[0] if contain else best)
        B = {"raw_rank": matched["raw_rank"], "candidate_rank": matched["candidate_rank"],
             "selected_top32": bool(matched["selected"]), "discovery_objectness": round(matched["discovery_score"], 4),
             "filtered_reason": matched["filtered_reason"], "matched_iou": round(iou(mb, matched["local_box"]), 3)}
        if matched["selected"] and matched.get("target_probability") is not None:
            cv = matched["class_scores"]
            C = {"target_probability": round(matched["target_probability"], 3),
                 "top3": [{"class": x["class"], "score": round(x["score"], 3)} for x in matched["top3"]],
                 "argmax_class": OBJECT_CLASSES[int(np.argmax(cv))], "argmax_score": round(float(max(cv)), 3),
                 "emitted": bool(matched["emitted"]), "pipeline_filtered_reason": matched["filtered_reason"],
                 "full_16class": {OBJECT_CLASSES[i]: round(float(cv[i]), 3) for i in range(16)}}
        else:
            C = {"note": "matched discovery candidate was NOT selected into top-32 -> Visual Expert never ran on it",
                 "emitted": False}
        res, _, _ = expert.classify(img, [mb])
        r = res[0]; ov = r["class_scores"]
        D = {"ORACLE_CROP_DIAGNOSTIC": "manual box fed directly to VE; never a runtime input; not an official score",
             "target_probability": round(r["target_probability"], 3), "state": r["state"],
             "top3": [{"class": x["class"], "score": round(x["score"], 3)} for x in r["top3"]],
             "argmax_class": OBJECT_CLASSES[int(np.argmax(ov))], "argmax_score": round(float(max(ov)), 3),
             "full_16class": {OBJECT_CLASSES[i]: round(float(ov[i]), 3) for i in range(16)}}
        v3m = sorted([(iou(mb, x["box"]), x) for x in v3], key=lambda z: z[0], reverse=True)
        E = {"best_v3_iou": round(v3m[0][0], 3) if v3m else 0.0,
             "best_v3_label": v3m[0][1]["label"] if v3m and v3m[0][0] > 0 else None,
             "caveat": "V3 is a model prediction, not ground truth"}
        fem = sorted([(iou(mb, f["box"]), f) for f in final], key=lambda z: z[0], reverse=True)
        emitted_overlap = {"best_emitted_iou": round(fem[0][0], 3) if fem else 0.0,
                           "best_emitted_label": fem[0][1]["label"] if fem and fem[0][0] > 0 else None,
                           "best_emitted_conf": round(fem[0][1]["conf"], 3) if fem and fem[0][0] > 0 else None}
        if best_iou < 0.10 and not contain:
            stage = "DISCOVERY_MISS (no raw proposal overlaps target)"
        elif not matched["selected"]:
            stage = "RANKING/BUDGET_DROP (raw proposal exists but not in top-32)"
        elif C.get("emitted") and emitted_overlap["best_emitted_iou"] >= 0.10:
            stage = "EMITTED (check class correctness)"
        elif matched["selected"] and matched.get("target_probability", 1.0) is not None and matched.get("target_probability", 1.0) < 0.5:
            stage = "FILTER_REJECT (target head <0.5 on discovery crop)"
        else:
            stage = "LOCALIZATION/OUTPUT_SUPPRESSION"
        report.append({"target": t["id"], "class_guess": t["class_guess"], "manual_confidence": t["manual_confidence"],
                       "manual_box": mb, "A_discovery": A, "B_selection": B, "C_discovery_crop_recognition": C,
                       "D_oracle_crop": D, "E_v3_reference": E, "final_emitted_overlap": emitted_overlap,
                       "diagnosed_stage": stage})
        pad = 70
        x1 = int(max(0, mb[0]-pad)); y1 = int(max(0, mb[1]-pad))
        x2 = int(min(960, mb[2]+pad)); y2 = int(min(540, mb[3]+pad))
        crop = img[y1:y2, x1:x2].copy(); mag = 6
        big = cv2.resize(crop, ((x2-x1)*mag, (y2-y1)*mag), interpolation=cv2.INTER_NEAREST)

        def rect(b, color, label, yoff=-4):
            cv2.rectangle(big, (int((b[0]-x1)*mag), int((b[1]-y1)*mag)), (int((b[2]-x1)*mag), int((b[3]-y1)*mag)), color, 2)
            if label:
                cv2.putText(big, label, (int((b[0]-x1)*mag), max(12, int((b[1]-y1)*mag)+yoff)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
        rect(matched["local_box"], (0, 255, 255), f"raw#{matched['raw_rank']} d={matched['discovery_score']:.2f}", -4)
        if matched["selected"]:
            rect(matched["local_box"], (255, 200, 0), f"sel#{matched['candidate_rank']}", 16)
        for f in final:
            if iou(mb, f["box"]) >= 0.05:
                rect(f["box"], (0, 0, 255), f"{f['label']} {f['conf']:.2f}", 34)
        rect(mb, (0, 255, 0), f"{t['id']}:{t['class_guess']}", -22)
        cv2.imwrite(f"{CLOSE}/{t['id']}.png", big)

json.dump({"note": "Human-directed target audit. Oracle crops are diagnostics, not official scores. V3 is a model output, not truth.",
           "targets": report}, open(f"{OUT}/target_audit.json", "w"), indent=2)
print("TARGET | conf | rawIoU | rawRank | selTop32 | tgtProb(disc) | discTop1 | oracleTop1(tgtProb,state) | emitted | stage")
for r in report:
    A = r["A_discovery"]; B = r["B_selection"]; C = r["C_discovery_crop_recognition"]; D = r["D_oracle_crop"]
    disc = f"{C['top3'][0]['class']}:{C['top3'][0]['score']}" if "top3" in C else "n/a(not-selected)"
    tgtp = C.get("target_probability", "-")
    orc = f"{D['top3'][0]['class']}:{D['top3'][0]['score']} tp={D['target_probability']}/{D['state']}"
    print(f"{r['target']} | {r['manual_confidence']} | {A['best_raw_iou']} | raw#{B['raw_rank']} | {B['selected_top32']} | {tgtp} | {disc} | {orc} | {C.get('emitted')} | {r['diagnosed_stage']}")
print("WROTE", f"{OUT}/target_audit.json")
