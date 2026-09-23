"""Offline ablation of the three V6 admission gates, kept SEPARATE from the detector comparison.

For each hosted observation we take the detector's post-merge candidates (production merge), run
the production DINOv2 + background head on every candidate, and then simulate gate settings:

  det_conf         admission threshold on the raw detector score
  classify_min_px  minimum box short side that is allowed to be classified at all
  target_min       background-head P(target) needed to open a track

Recovered = a frozen manual target has a passing candidate at IoU >= 0.5 (tight or 1.25-padded).
Burden    = passing candidates that do NOT overlap any frozen target (IoU < 0.2): the false-positive
            load the rest of the pipeline would have to carry. The 20 frames are a diagnostic
            sample, so burden is indicative, not a sequence-level FP rate.
"""
import json, glob, sys, itertools, hashlib
import numpy as np, cv2
from ultralytics import YOLO
sys.path.insert(0, "/workspace/v6-diagnostic-release-8ae3b97")
from v5.gpu.discovery_v54 import RejectingExpert

AUD = "/workspace/audit-l1l2"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/training/ms1/weights/best.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{AUD}/gate_ablation.json"
PAD = 1.25


def iou(a, b):
    ix = max(0, min(a[2], b[2])-max(a[0], b[0])); iy = max(0, min(a[3], b[3])-max(a[1], b[1])); i = ix*iy
    u = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u > 0 else 0.


def padb(b, s):
    cx, cy, w, h = (b[0]+b[2])/2, (b[1]+b[3])/2, (b[2]-b[0])*s, (b[3]-b[1])*s
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


def merge(props, thr=0.6):                       # identical to V6Pipeline._merge_props
    props = sorted(props, key=lambda p: p["score"], reverse=True); kept = []
    for p in props:
        if all(iou(p["box"], q["box"]) <= thr for q in kept): kept.append(p)
    return kept


ann = json.load(open(f"{AUD}/manual_annotations_frozen.json"))
tg = {}
for t in ann["targets"]: tg.setdefault((t["frame_index"], t["level"]), []).append(t)
model = YOLO(CKPT); expert = RejectingExpert("/workspace/assets", device="cuda:0")
model.predict(np.zeros((540, 960, 3), np.uint8), imgsz=960, conf=.5, verbose=False, device=0)

DET_CONFS = [0.15, 0.10, 0.05]
MIN_PX = [22, 16, 12]
TMIN = [0.50, 0.45, 0.40]
rows = []
for f in sorted(glob.glob(f"{AUD}/captures/*.png")):
    sc = json.load(open(f[:-4]+".json")); fi = sc["frame_index"]; lv = sc["received_camera"]["resolution_level"]
    im = cv2.imread(f)
    r = model.predict(im, conf=min(DET_CONFS), imgsz=960, iou=0.7, max_det=300, verbose=False, device=0)[0]
    props = [{"box": [float(v) for v in b], "score": float(c)}
             for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    for dc in DET_CONFS:
        cands = merge([p for p in props if p["score"] >= dc])
        if not cands: continue
        boxes = [c["box"] for c in cands]
        res, _, _ = expert.classify(im, boxes)
        for c, q in zip(cands, res):
            b = c["box"]; ms = min(b[2]-b[0], b[3]-b[1])
            best_t, best_i = None, 0.
            for t in tg.get((fi, lv), []):
                v = max(iou(b, t["box_local_xyxy"]), iou(b, padb(t["box_local_xyxy"], PAD)))
                if v > best_i: best_i, best_t = v, t["target_id"]
            rows.append(dict(frame=fi, level=lv, det_conf=dc, score=c["score"], min_side=ms,
                             target_p=float(q["target_probability"]), top1=q["top3"][0]["class"],
                             best_target=best_t, best_iou=best_i))
json.dump(rows, open(OUT, "w"), indent=1)

targets = [t["target_id"] for t in ann["targets"]]
print(f'{"det_conf":>8} {"min_px":>6} {"tgt_min":>7} | {"recovered/14":>12} {"distinct":>8} | {"FP-burden/frame":>15} | recovered ids')
base = None
for dc, mp, tm in itertools.product(DET_CONFS, MIN_PX, TMIN):
    ok, fp = set(), 0
    for r in rows:
        if r["det_conf"] != dc: continue
        if r["min_side"] < mp: continue                      # never classified -> never emitted
        if r["target_p"] < tm: continue                      # background head rejects
        if r["best_iou"] >= 0.5 and r["best_target"]: ok.add(r["best_target"])
        elif r["best_iou"] < 0.2: fp += 1
    grp = {t.rsplit("_", 1)[0] for t in ok}
    line = f'{dc:>8} {mp:>6} {tm:>7} | {len(ok):>7}/14     {len(grp):>8} | {fp/20:>15.1f} | {",".join(sorted(ok))}'
    if (dc, mp, tm) == (0.15, 22, 0.50): base = line; line += "   <== CURRENT PRODUCTION"
    print(line)
