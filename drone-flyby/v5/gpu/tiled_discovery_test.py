import os, json, time, cv2, numpy as np
from v5.discovery import Discovery
from v2.geometry import iou as _iou

ASSETS = os.environ.get("V5_ASSETS", "/workspace/assets")
disc = Discovery(ASSETS, budget=256, device=os.environ.get("V5_DEVICE", "cuda:0"), threads=2)
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


def tiled_pass(img, cols, rows, overlap=0.25, thr=None):
    """Run disc._pass over an overlapping grid; return merged raw boxes (full-image coords)."""
    H, W = img.shape[:2]
    tw, th = W/cols, H/rows
    ox, oy = tw*overlap, th*overlap
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(max(0, c*tw - ox)); y0 = int(max(0, r*th - oy))
            x1 = int(min(W, (c+1)*tw + ox)); y1 = int(min(H, (r+1)*th + oy))
            tile = img[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            if thr is not None:
                old = disc.threshold; disc.threshold = thr
            out.extend(disc._pass(tile, offset=(x0, y0), source=f"tile{cols}x{rows}"))
            if thr is not None:
                disc.threshold = old
    return out


def recall_report(name, get_boxes):
    print(f"\n=== {name} ===")
    per = {}
    for cap, fn in files.items():
        img = cv2.imread(f"/workspace/hosted-v3/images/{fn}")
        t = time.perf_counter()
        boxes = get_boxes(img)
        ms = (time.perf_counter()-t)*1000
        for tg in by_cap[cap]:
            mb = tg["box"]
            best = max((iou(mb, b["local_box"]) for b in boxes), default=0.0)
            per[tg["id"]] = best
            flag = "  <== IoU>=0.5" if best >= 0.5 else ("  (>=0.3)" if best >= 0.3 else "")
            print(f"  {tg['id']:14s} bestIoU={best:.3f}{flag}")
        print(f"  [{cap}] n_boxes={len(boxes)} time={ms:.0f}ms")
    hit = sum(1 for v in per.values() if v >= 0.5)
    print(f"  >>> targets with IoU>=0.5: {hit}/{len(per)}")
    return per


# Baseline: full frame only
recall_report("A. full-frame only (baseline)", lambda im: disc._pass(im))
# Built-in-style 4 quadrant tiles
recall_report("B. 2x2 tiles overlap0.25", lambda im: disc._pass(im) + tiled_pass(im, 2, 2, 0.25))
# Finer 4x3 tiles
recall_report("C. full + 4x3 tiles overlap0.3", lambda im: disc._pass(im) + tiled_pass(im, 4, 3, 0.30))
# Aggressive 6x4 tiles + lower threshold on tiles
recall_report("D. full + 6x4 tiles overlap0.3 thr0.02", lambda im: disc._pass(im) + tiled_pass(im, 6, 4, 0.30, thr=0.02))
