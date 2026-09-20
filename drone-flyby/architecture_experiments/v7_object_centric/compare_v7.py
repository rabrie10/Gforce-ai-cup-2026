"""Decisive experiment: ms1 (baseline) vs V7 candidate on the FROZEN hosted L1/L2 annotations.

Evaluation-only use of the hosted diagnostic captures. Manual boxes are read verbatim from
manual_annotations_frozen.json (sha256 0dc88cb6...); nothing here modifies them.

Padding sensitivity is PREDECLARED: IoU is reported against (a) the frozen tight box and
(b) that box scaled by 1.25 isotropically, chosen before any candidate result was seen,
because the audit showed official Helsinki boxes are looser than tight object extents.
"""
import json, glob, sys, time, hashlib, os
import numpy as np, cv2
from ultralytics import YOLO

AUD = "/workspace/audit-l1l2"
PAD = 1.25                     # predeclared padding-sensitivity factor
BASE = "/workspace/training/ms1/weights/best.pt"
CAND = sys.argv[1] if len(sys.argv) > 1 else "/workspace/training/v7a/weights/best.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/audit-l1l2/v7_compare.json"

# distinct physical objects (same object seen in consecutive observations)
GROUP = {"f52_heli": "heliA", "f55_heli": "heliA", "f6_armor": "armorA", "f7_armor": "armorA",
         "f50_armor": "armorB", "f51_armor": "armorB", "f100_armor": "armorC", "f101_armor": "armorC",
         "f105_armor": "armorC", "f204_veh_small": "vehD", "f204_launcher_large": "launcherE",
         "f4_armor": "armorF", "f50_dark_b": "darkG", "f50_dark_a": "darkH"}


def iou(a, b):
    ix = max(0, min(a[2], b[2])-max(a[0], b[0])); iy = max(0, min(a[3], b[3])-max(a[1], b[1])); i = ix*iy
    u = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u > 0 else 0.0


def pad(b, s):
    cx, cy, w, h = (b[0]+b[2])/2, (b[1]+b[3])/2, (b[2]-b[0])*s, (b[3]-b[1])*s
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


ann = json.load(open(f"{AUD}/manual_annotations_frozen.json"))
print("frozen annotations sha256:", hashlib.sha256(open(f"{AUD}/manual_annotations_frozen.json", "rb").read()).hexdigest())
imgs = {}
for f in sorted(glob.glob(f"{AUD}/captures/*.png")):
    sc = json.load(open(f[:-4]+".json"))
    imgs[(sc["frame_index"], sc["received_camera"]["resolution_level"])] = (cv2.imread(f), sc)

res = {"baseline": BASE, "candidate": CAND, "pad_factor": PAD, "targets": [], "frames": {}, "models": {}}
for tag, w in (("baseline", BASE), ("candidate", CAND)):
    res["models"][tag] = dict(path=w, sha256=hashlib.sha256(open(w, "rb").read()).hexdigest())
    m = YOLO(w)
    m.predict(np.zeros((540, 960, 3), np.uint8), imgsz=960, conf=.5, verbose=False, device=0)  # warmup
    for (fi, lv), (im, sc) in imgs.items():
        t = time.perf_counter()
        r = m.predict(im, conf=0.001, imgsz=960, iou=0.7, max_det=300, verbose=False, device=0)[0]
        ms = (time.perf_counter()-t)*1000
        props = [{"box": [float(v) for v in b], "score": float(c)}
                 for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        t2 = time.perf_counter()
        rp = m.predict(im, conf=0.15, imgsz=960, verbose=False, device=0)[0]
        ms_prod = (time.perf_counter()-t2)*1000
        k = f"{fi}_{lv}"
        res["frames"].setdefault(k, {})[tag] = dict(
            n_all=len(props), n_conf15=int((r.boxes.conf.cpu().numpy() >= .15).sum()),
            n_conf05=int((r.boxes.conf.cpu().numpy() >= .05).sum()), n_prod=len(rp.boxes),
            latency_ms_full=round(ms, 1), latency_ms_prod=round(ms_prod, 1),
            props=[p for p in props[:64]])
for t in ann["targets"]:
    fi, lv, gt = t["frame_index"], t["level"], t["box_local_xyxy"]
    row = dict(target=t["target_id"], group=GROUP[t["target_id"]], frame=fi, level=lv,
               family=t["family"], difficulty=t["difficulty"], manual_conf=t["existence_confidence"],
               px=f'{t["px_w"]}x{t["px_h"]}', short_side=t["short_side"])
    for tag in ("baseline", "candidate"):
        fr = res["frames"][f"{fi}_{lv}"][tag]
        P = fr["props"]
        for label, g in (("", gt), ("_pad", pad(gt, PAD))):
            ious = [iou(g, p["box"]) for p in P]
            o = int(np.argmax(ious)) if ious else None
            row[f"{tag}{label}_best_iou"] = round(max(ious), 3) if ious else 0.0
            row[f"{tag}{label}_best_rank"] = o
            row[f"{tag}{label}_best_score"] = round(P[o]["score"], 4) if o is not None else None
            row[f"{tag}{label}_rank_first50"] = next((i for i, v in enumerate(ious) if v >= .5), None)
            for kk in (16, 32, 64):
                row[f"{tag}{label}_best_iou_top{kk}"] = round(max(ious[:kk]+[0]), 3)
        row[f"{tag}_n_conf15"] = fr["n_conf15"]
    res["targets"].append(row)
json.dump(res, open(OUT, "w"), indent=1)

hdr = f'{"target":20s} {"grp":9s} {"diff":8s} {"px":8s} | base tight/pad  | cand tight/pad  | base>=.5 cand>=.5'
print(hdr); print("-"*len(hdr))
for r in res["targets"]:
    print(f'{r["target"]:20s} {r["group"]:9s} {r["difficulty"]:8s} {r["px"]:8s} | '
          f'{r["baseline_best_iou"]:.2f}/{r["baseline_pad_best_iou"]:.2f} r{str(r["baseline_best_rank"]):>3s} | '
          f'{r["candidate_best_iou"]:.2f}/{r["candidate_pad_best_iou"]:.2f} r{str(r["candidate_best_rank"]):>3s} | '
          f'{str(r["baseline_rank_first50"]):>5s} {str(r["candidate_rank_first50"]):>5s}')
for tag in ("baseline", "candidate"):
    n50 = sum(r[f"{tag}_best_iou"] >= .5 for r in res["targets"])
    n50p = sum(r[f"{tag}_pad_best_iou"] >= .5 for r in res["targets"])
    n30 = sum(max(r[f"{tag}_best_iou"], r[f"{tag}_pad_best_iou"]) >= .3 for r in res["targets"])
    t32 = sum(r[f"{tag}_best_iou_top32"] >= .5 for r in res["targets"])
    grp = len({r["group"] for r in res["targets"] if max(r[f"{tag}_best_iou"], r[f"{tag}_pad_best_iou"]) >= .5})
    burden = np.mean([res["frames"][k][tag]["n_conf15"] for k in res["frames"]])
    lat = np.mean([res["frames"][k][tag]["latency_ms_prod"] for k in res["frames"]])
    print(f'{tag:9s} IoU>=.5 tight {n50}/14 pad {n50p}/14 | IoU>=.3 {n30}/14 | top32@.5 {t32}/14 | '
          f'distinct objects @.5 {grp} | mean conf>=.15 props {burden:.1f} | prod latency {lat:.0f}ms')
