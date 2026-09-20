import os, json, cv2, numpy as np
from ultralytics import YOLO, YOLOWorld
files = {60: "00060_80324c03bcc65ff1d04c510e_180324c03bcc65ff1d04c510e_60_0_1920_1080.png",
         125: "00125_80324c03bcc65ff1d04c510e_80324c03bcc65ff1d04c510e_125_0_1920_1080.png"}
manual = json.load(open("/workspace/results/manual_targets.json"))["targets"]
by_cap = {}
for t in manual:
    by_cap.setdefault(t["capture"], []).append(t)
OUT = "/workspace/results/viz_v54"; os.makedirs(OUT, exist_ok=True)
PROMPTS = ["airplane", "aircraft", "helicopter", "hangar", "military vehicle", "tank", "tower", "boat", "ship", "building"]
obb = YOLO("/workspace/assets/yolo11l-obb.pt")
world = YOLOWorld("/workspace/assets/yolov8l-worldv2.pt"); world.set_classes(PROMPTS)


def tiles(W, H, cols, rows, ov=0.25):
    tw, th = W/cols, H/rows; ox, oy = tw*ov, th*ov
    for r in range(rows):
        for c in range(cols):
            yield (int(max(0, c*tw-ox)), int(max(0, r*th-oy)), int(min(W, (c+1)*tw+ox)), int(min(H, (r+1)*th+oy)))


def obb_dets(img, imgsz):
    r = obb.predict(img, conf=0.08, imgsz=imgsz, verbose=False, device=0)[0]
    out = []
    if r.obb is not None and len(r.obb) > 0:
        cs = r.obb.xyxyxyxy.cpu().numpy(); cls = r.obb.cls.cpu().numpy(); cf = r.obb.conf.cpu().numpy()
        names = r.names
        for poly, c, f in zip(cs, cls, cf):
            xs, ys = poly[:, 0], poly[:, 1]
            out.append(([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], f"{names[int(c)]}{f:.2f}"))
    return out


def world_dets(img, imgsz):
    r = world.predict(img, conf=0.02, imgsz=imgsz, verbose=False, device=0)[0]
    out = []
    if r.boxes is not None and len(r.boxes) > 0:
        for b, c, f in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append(([float(b[0]), float(b[1]), float(b[2]), float(b[3])], f"{PROMPTS[int(c)]}{f:.2f}"))
    return out


def all_dets(img, fn, imf, imt):
    H, W = img.shape[:2]
    d = [(b, l, 'f') for b, l in fn(img, imf)]
    for (x0, y0, x1, y1) in tiles(W, H, 3, 2):
        t = img[y0:y1, x0:x1]
        for b, l in fn(t, imt):
            d.append(([b[0]+x0, b[1]+y0, b[2]+x0, b[3]+y0], l, 't'))
    return d


for cap, fnm in files.items():
    img = cv2.imread(f"/workspace/hosted-v3/images/{fnm}")
    obbd = all_dets(img, obb_dets, 1024, 640)
    wd = all_dets(img, world_dets, 1024, 640)
    for tg in by_cap[cap]:
        mb = tg["box"]; pad = 60
        x1 = int(max(0, mb[0]-pad)); y1 = int(max(0, mb[1]-pad)); x2 = int(min(960, mb[2]+pad)); y2 = int(min(540, mb[3]+pad))
        mag = 6; crop = img[y1:y2, x1:x2].copy()
        big = cv2.resize(crop, ((x2-x1)*mag, (y2-y1)*mag), interpolation=cv2.INTER_NEAREST)

        def rect(b, color, lab, yo):
            X1, Y1, X2, Y2 = [int((b[0]-x1)*mag), int((b[1]-y1)*mag), int((b[2]-x1)*mag), int((b[3]-y1)*mag)]
            cv2.rectangle(big, (X1, Y1), (X2, Y2), color, 2)
            if lab:
                cv2.putText(big, lab, (X1, max(10, Y1+yo)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
        for b, l, s in obbd:
            if b[2] > x1 and b[0] < x2 and b[3] > y1 and b[1] < y2:
                rect(b, (0, 220, 255), l, -3)
        for b, l, s in wd:
            if b[2] > x1 and b[0] < x2 and b[3] > y1 and b[1] < y2:
                rect(b, (255, 200, 0), l, 12)
        rect(mb, (0, 255, 0), tg["id"], -16)
        cv2.imwrite(f"{OUT}/{tg['id']}.png", big)
print("done", OUT)
