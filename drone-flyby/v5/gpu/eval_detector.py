"""Evaluate a detector (via its assets dir with discovery.onnx) on the 8 hosted
manual targets: best raw IoU and IoU>=0.5, full-frame and tiled. Diagnostic only."""
import os, sys, json, cv2, numpy as np
from v5.discovery import Discovery

ASSETS = sys.argv[1] if len(sys.argv) > 1 else "/workspace/assets"
NAME = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(ASSETS)
disc = Discovery(ASSETS, budget=256, device="cuda:0", threads=2)
files = {60: "00060_80324c03bcc65ff1d04c510e_180324c03bcc65ff1d04c510e_60_0_1920_1080.png",
         125: "00125_80324c03bcc65ff1d04c510e_80324c03bcc65ff1d04c510e_125_0_1920_1080.png"}
manual = json.load(open("/workspace/results/manual_targets.json"))["targets"]
by_cap = {}
for t in manual:
    by_cap.setdefault(t["capture"], []).append(t)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.


def tiled(img, cols, rows, overlap=0.3):
    H, W = img.shape[:2]; tw, th = W/cols, H/rows; ox, oy = tw*overlap, th*overlap
    out = list(disc._pass(img))
    for r in range(rows):
        for c in range(cols):
            x0 = int(max(0, c*tw-ox)); y0 = int(max(0, r*th-oy))
            x1 = int(min(W, (c+1)*tw+ox)); y1 = int(min(H, (r+1)*th+oy))
            t = img[y0:y1, x0:x1]
            if t.size:
                out.extend(disc._pass(t, offset=(x0, y0), source="tile"))
    return out


print(f"===== detector: {NAME} =====")
res = {}
for mode, getter in [("full", lambda im: disc._pass(im)), ("full+4x3tiled", lambda im: tiled(im, 4, 3))]:
    hits = 0; rows = {}
    for cap, fn in files.items():
        img = cv2.imread(f"/workspace/hosted-v3/images/{fn}")
        boxes = getter(img)
        for t in by_cap[cap]:
            best = max((iou(t["box"], b["local_box"]) for b in boxes), default=0.0)
            rows[t["id"]] = round(best, 3)
            if best >= 0.5:
                hits += 1
    print(f"  [{mode}] IoU>=0.5: {hits}/8 | " + " ".join(f"{k}={v}" for k, v in rows.items()))
    res[mode] = {"hits_ge_0.5": hits, "per_target": rows}
json.dump({"detector": NAME, "result": res}, open(f"/workspace/results/eval_{NAME}.json", "w"), indent=2)
