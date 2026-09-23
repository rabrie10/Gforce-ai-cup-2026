import os, json, time, cv2, numpy as np
from ultralytics import YOLO, YOLOWorld

files = {60: "00060_80324c03bcc65ff1d04c510e_180324c03bcc65ff1d04c510e_60_0_1920_1080.png",
         125: "00125_80324c03bcc65ff1d04c510e_80324c03bcc65ff1d04c510e_125_0_1920_1080.png"}
manual = json.load(open("/workspace/results/manual_targets.json"))["targets"]
by_cap = {}
for t in manual:
    by_cap.setdefault(t["capture"], []).append(t)
PROMPTS = ["airplane", "aircraft", "helicopter", "hangar", "military vehicle", "tank",
           "truck", "tower", "radar dish", "missile launcher", "boat", "ship", "building"]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.


def tiles(W, H, cols, rows, ov=0.25):
    tw, th = W/cols, H/rows; ox, oy = tw*ov, th*ov
    for r in range(rows):
        for c in range(cols):
            yield (int(max(0, c*tw-ox)), int(max(0, r*th-oy)), int(min(W, (c+1)*tw+ox)), int(min(H, (r+1)*th+oy)))


def obb_boxes(model, img, conf=0.08, imgsz=1024):
    out = []
    r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False, device=0)[0]
    if r.obb is not None and len(r.obb) > 0:
        corners = r.obb.xyxyxyxy.cpu().numpy()  # (N,4,2)
        for poly in corners:
            xs, ys = poly[:, 0], poly[:, 1]
            out.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    return out


def world_boxes(model, img, conf=0.02, imgsz=1024):
    out = []
    r = model.predict(img, conf=conf, imgsz=imgsz, verbose=False, device=0)[0]
    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes.xyxy.cpu().numpy():
            out.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    return out


def run(model, img, boxfn, imgsz_full, imgsz_tile, tiled):
    boxes = boxfn(model, img, imgsz=imgsz_full)
    if tiled:
        H, W = img.shape[:2]
        for (x0, y0, x1, y1) in tiles(W, H, 3, 2, 0.25):
            t = img[y0:y1, x0:x1]
            for b in boxfn(model, t, imgsz=imgsz_tile):
                boxes.append([b[0]+x0, b[1]+y0, b[2]+x0, b[3]+y0])
    return boxes


obb = YOLO("/workspace/assets/yolo11l-obb.pt")
world = YOLOWorld("/workspace/assets/yolov8l-worldv2.pt")
world.set_classes(PROMPTS)
imgs = {cap: cv2.imread(f"/workspace/hosted-v3/images/{fn}") for cap, fn in files.items()}
# warmup
_ = obb_boxes(obb, imgs[60]); _ = world_boxes(world, imgs[60])

report = {}
for label, fn2, ft, imf, imt in [
        ("OBB full", obb_boxes, False, 1024, 640),
        ("OBB tiled3x2", obb_boxes, True, 1024, 640),
        ("WORLD full", world_boxes, False, 1024, 640),
        ("WORLD tiled3x2", world_boxes, True, 1024, 640)]:
    model = obb if "OBB" in label else world
    print(f"\n=== {label} ===")
    hits = 0; per = {}; tsum = 0; nsum = 0
    for cap in (60, 125):
        t = time.perf_counter()
        boxes = run(model, imgs[cap], fn2, imf, imt, ft)
        tsum += (time.perf_counter()-t)*1000; nsum += len(boxes)
        for tg in by_cap[cap]:
            best = max((iou(tg["box"], b) for b in boxes), default=0.0)
            per[tg["id"]] = round(best, 3)
            if best >= 0.5:
                hits += 1
    print(f"  IoU>=0.5: {hits}/8 | n_boxes~{nsum} time~{tsum:.0f}ms")
    print("  " + " ".join(f"{k}={v}" for k, v in per.items()))
    report[label] = {"hits": hits, "per_target": per, "n_boxes": nsum, "ms": round(tsum, 0)}

# UNION of OBB tiled + WORLD tiled
print("\n=== UNION (OBB tiled + WORLD tiled) ===")
hits = 0; per = {}
for cap in (60, 125):
    b1 = run(obb, imgs[cap], obb_boxes, 1024, 640, True)
    b2 = run(world, imgs[cap], world_boxes, 1024, 640, True)
    boxes = b1 + b2
    for tg in by_cap[cap]:
        best = max((iou(tg["box"], b) for b in boxes), default=0.0)
        per[tg["id"]] = round(best, 3)
        if best >= 0.5:
            hits += 1
print(f"  IoU>=0.5: {hits}/8")
print("  " + " ".join(f"{k}={v}" for k, v in per.items()))
report["UNION_tiled"] = {"hits": hits, "per_target": per}
json.dump(report, open("/workspace/results/detector_v54_test.json", "w"), indent=2)
print("\nWROTE /workspace/results/detector_v54_test.json")
