import os, json, cv2, numpy as np
from v5.discovery import Discovery

ASSETS = os.environ.get("V5_ASSETS", "/workspace/assets")
disc = Discovery(ASSETS, budget=256, device=os.environ.get("V5_DEVICE", "cuda:0"), threads=2)
files = {60: "00060_80324c03bcc65ff1d04c510e_180324c03bcc65ff1d04c510e_60_0_1920_1080.png",
         125: "00125_80324c03bcc65ff1d04c510e_80324c03bcc65ff1d04c510e_125_0_1920_1080.png"}
REGION = [0, 0, 3840, 2160]
manual = json.load(open("/workspace/results/manual_targets.json"))["targets"]
by_cap = {}
for t in manual:
    by_cap.setdefault(t["capture"], []).append(t)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.


print("target | conf | bestIoU | anyIoU>=0.5? | candRank@bestQual | survive@32/64/128/256 | rawTotal")
rows = []
for cap, fn in files.items():
    img = cv2.imread(f"/workspace/hosted-v3/images/{fn}")
    raw, sel, stages = disc.propose(img, REGION)
    for t in by_cap[cap]:
        mb = t["box"]
        scored = sorted(((iou(mb, c["local_box"]), c) for c in raw), key=lambda z: z[0], reverse=True)
        best_iou = scored[0][0]
        qual = [(i, c) for i, c in scored if i >= 0.5 and c["candidate_rank"] is not None]
        if qual:
            qi, qc = qual[0]
            crank = qc["candidate_rank"]
            surv = {b: (crank <= b) for b in (32, 64, 128, 256)}
            survstr = "/".join("Y" if surv[b] else "n" for b in (32, 64, 128, 256))
            qstr = f"{crank}(IoU{qi:.2f})"
        else:
            survstr = "-/-/-/-"; qstr = "none>=0.5"
        print(f"{t['id']} | {t['manual_confidence'][:10]} | {best_iou:.3f} | {'YES' if best_iou>=0.5 else 'no'} | {qstr} | {survstr} | {len(raw)}")
        rows.append({"target": t["id"], "best_iou": round(best_iou, 3), "any_iou_ge_0.5": best_iou >= 0.5,
                     "best_qualifying_candidate_rank": (qual[0][1]["candidate_rank"] if qual else None),
                     "survive_32": bool(qual and qual[0][1]["candidate_rank"] <= 32),
                     "survive_64": bool(qual and qual[0][1]["candidate_rank"] <= 64),
                     "survive_128": bool(qual and qual[0][1]["candidate_rank"] <= 128),
                     "survive_256": bool(qual and qual[0][1]["candidate_rank"] <= 256),
                     "raw_total": len(raw)})
    print(f"  [{cap}] raw={len(raw)} post_nms={stages['nms_count']}")
json.dump({"detector": "epoch30-final", "rows": rows}, open("/workspace/results/discovery_final_audit.json", "w"), indent=2)
print("WROTE /workspace/results/discovery_final_audit.json")
