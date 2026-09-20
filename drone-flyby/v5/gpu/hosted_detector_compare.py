"""Measure hosted-domain DETECTION: does the V6 detector (ms1, Helsinki fine-tuned) propose the
manual hosted targets on captures 60/125? Compare vs pretrained OBB + YOLO-World. Diagnostic only."""
import os, json, cv2, numpy as np
from ultralytics import YOLO
from v5.gpu.discovery_v54 import MergedDiscovery
from v5.pipeline import Config

cfg = Config()
ms1 = YOLO("/workspace/training/ms1/weights/best.pt")
merged = MergedDiscovery(cfg.assets, budget=256, device=cfg.device)  # OBB + World
files = {60: "00060_80324c03bcc65ff1d04c510e_180324c03bcc65ff1d04c510e_60_0_1920_1080.png",
         125: "00125_80324c03bcc65ff1d04c510e_80324c03bcc65ff1d04c510e_125_0_1920_1080.png"}
manual = json.load(open("/workspace/results/manual_targets.json"))["targets"]
mby = {}
for t in manual:
    mby.setdefault(t["capture"], []).append(t)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.


def tiles(W, H, c, r, ov=0.3):
    tw, th = W/c, H/r; ox, oy = tw*ov, th*ov
    for ry in range(r):
        for cx in range(c):
            yield (int(max(0, cx*tw-ox)), int(max(0, ry*th-oy)), int(min(W, (cx+1)*tw+ox)), int(min(H, (ry+1)*th+oy)))


def ms1_boxes(img, tiled):
    out = []
    def run(sub, ox, oy):
        rr = ms1.predict(sub, conf=0.12, imgsz=960, verbose=False, device=0)[0]
        if rr.boxes is not None:
            for b in rr.boxes.xyxy.cpu().numpy():
                out.append([b[0]+ox, b[1]+oy, b[2]+ox, b[3]+oy])
    run(img, 0, 0)
    if tiled:
        H, W = img.shape[:2]
        for x0, y0, x1, y1 in tiles(W, H, 3, 2):
            run(img[y0:y1, x0:x1], x0, y0)
    return out


def merged_src(img):  # OBB+World proposals in source coords (full frame region)
    raw, sel, _ = merged.propose(img, [0, 0, 3840, 2160])
    return [c["source_box"] for c in raw]  # note: these are already source-scaled from 960x540 view


print("target | ms1_full | ms1_tiled | OBB+World | union(ms1_tiled+OBBWorld)  [bestIoU]")
for cap, fn in files.items():
    img = cv2.imread(f"/workspace/hosted-v3/images/{fn}")  # 960x540 L0
    mf = ms1_boxes(img, False); mt = ms1_boxes(img, True)
    ow = [[b[0]/4, b[1]/4, b[2]/4, b[3]/4] for b in merged_src(img)]  # source->L0(960x540) for IoU vs manual (manual in L0)
    for t in mby[cap]:
        gb = t["box"]
        i_mf = max((iou(gb, b) for b in mf), default=0)
        i_mt = max((iou(gb, b) for b in mt), default=0)
        i_ow = max((iou(gb, b) for b in ow), default=0)
        i_un = max(i_mt, i_ow)
        print(f"{t['id']:14s} {i_mf:.2f}      {i_mt:.2f}      {i_ow:.2f}      {i_un:.2f}")
    print(f"  [{cap}] ms1_full_n={len(mf)} ms1_tiled_n={len(mt)} obbworld_n={len(ow)}")
