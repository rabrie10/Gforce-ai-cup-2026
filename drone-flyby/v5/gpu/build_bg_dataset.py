"""Supervised target/background dataset on frozen DINOv2 features.
POS = genuine Helsinki target crops (dataset_v52). NEG = Helsinki background +
hosted boat/harbor/vehicle crops (DOTA-detected) from frames OTHER than 60/125.
Frames 60/125 and the 8 manual boxes are NEVER used for training (diagnostics only).
Features extracted exactly as runtime (VisualExpert crop+encoder)."""
import os, glob, json, cv2, numpy as np
from ultralytics import YOLO
from v5.pipeline import Config
from v5.visual_expert import VisualExpert

cfg = Config()
expert = VisualExpert(cfg.assets, device=cfg.device)
obb = YOLO(f"{cfg.assets}/yolo11l-obb.pt")
D = "/workspace/assets/dataset_v52"
HOSTED = "/workspace/hosted-v3/images"
rng = np.random.default_rng(20260919)
NEG_DOTA = {"ship", "harbor", "large vehicle", "small vehicle", "bridge", "storage tank", "roundabout",
            "large-vehicle", "small-vehicle", "storage-tank"}
DIAG_FRAMES = {"00060", "00125"}


def feats_for(image, boxes):
    if not boxes:
        return np.empty((0, 384), np.float32)
    _, f, _ = expert.classify(image, boxes)
    return f


def denorm(line):
    _, cx, cy, w, h = map(float, line.split())
    return [(cx-w/2)*960, (cy-h/2)*540, (cx+w/2)*960, (cy+h/2)*540]


def jitter(b, n=2):
    out = [b]
    w, h = b[2]-b[0], b[3]-b[1]
    for _ in range(n):
        dx, dy = rng.uniform(-.15, .15)*w, rng.uniform(-.15, .15)*h
        sw, sh = rng.uniform(.85, 1.2), rng.uniform(.85, 1.2)
        cx, cy = (b[0]+b[2])/2+dx, (b[1]+b[3])/2+dy
        nw, nh = w*sw/2, h*sh/2
        nb = [max(0, cx-nw), max(0, cy-nh), min(960, cx+nw), min(540, cy+nh)]
        if nb[2]-nb[0] > 4 and nb[3]-nb[1] > 4:
            out.append(nb)
    return out


def helsinki(split):
    X, y = [], []
    for img_path in sorted(glob.glob(f"{D}/images/{split}/*.png")):
        lab = f"{D}/labels/{split}/" + os.path.basename(img_path)[:-4] + ".txt"
        im = cv2.imread(img_path)
        tboxes = [denorm(l) for l in open(lab).read().splitlines() if l.strip()]
        pos = []
        for b in tboxes:
            pos += jitter(b, 2 if split == "train" else 0)
        f = feats_for(im, pos)
        X.append(f); y += [1]*len(f)
        # background negatives avoiding targets
        negs = []
        for _ in range(10 if split == "train" else 4):
            for _t in range(20):
                s = int(rng.choice([28, 44, 70, 110]))
                x0 = int(rng.integers(0, 960-s)); y0 = int(rng.integers(0, 540-s))
                g = (x0-16, y0-16, x0+s+16, y0+s+16)
                if not any(min(g[2], b[2]) > max(g[0], b[0]) and min(g[3], b[3]) > max(g[1], b[1]) for b in tboxes):
                    negs.append([x0, y0, x0+s, y0+s]); break
        f = feats_for(im, negs)
        X.append(f); y += [0]*len(f)
    return np.concatenate(X), np.array(y)


def hosted_boats(exclude_diag=True, only_frames=None):
    X = []
    prov = []
    for p in sorted(glob.glob(f"{HOSTED}/*.png")):
        key = os.path.basename(p)[:5]
        if exclude_diag and key in DIAG_FRAMES:
            continue
        if only_frames is not None and key not in only_frames:
            continue
        im = cv2.imread(p)
        r = obb.predict(im, conf=0.30, imgsz=1280, verbose=False, device=0)[0]
        boxes = []
        if r.obb is not None and len(r.obb) > 0:
            names = r.names
            cs = r.obb.xyxyxyxy.cpu().numpy(); cl = r.obb.cls.cpu().numpy()
            for poly, c in zip(cs, cl):
                nm = names[int(c)].lower()
                if nm in NEG_DOTA:
                    xs, ys = poly[:, 0], poly[:, 1]
                    boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        if boxes:
            X.append(feats_for(im, boxes)); prov += [key]*len(boxes)
    return (np.concatenate(X) if X else np.empty((0, 384), np.float32)), prov


# reserve two hosted marina frames for a negative holdout (not 60/125)
allframes = sorted({os.path.basename(p)[:5] for p in glob.glob(f"{HOSTED}/*.png")} - DIAG_FRAMES)
holdout_neg_frames = set(allframes[-2:])
train_neg_frames = set(allframes) - holdout_neg_frames

Xh, yh = helsinki("train")
Xb, pb = hosted_boats(only_frames=train_neg_frames)
Xtr = np.concatenate([Xh, Xb]); ytr = np.concatenate([yh, np.zeros(len(Xb))])
# holdout
Xt, yt = helsinki("test")
Xbh, pbh = hosted_boats(only_frames=holdout_neg_frames)
Xho = np.concatenate([Xt, Xbh]); yho = np.concatenate([yt, np.zeros(len(Xbh))])

np.savez("/workspace/assets/bg_head_data.npz", Xtr=Xtr, ytr=ytr, Xho=Xho, yho=yho)
prov = {"train_pos_helsinki": int(yh.sum()), "train_neg_helsinki_bg": int((yh == 0).sum()),
        "train_neg_hosted_boats": int(len(Xb)), "train_neg_hosted_frames": sorted(train_neg_frames),
        "holdout_pos_helsinki_test": int(yt.sum()), "holdout_neg_helsinki_bg": int((yt == 0).sum()),
        "holdout_neg_hosted_boats": int(len(Xbh)), "holdout_neg_hosted_frames": sorted(holdout_neg_frames),
        "caveats": ["Helsinki positives are the SAME physical instances across frames; frame split is not independent object generalization.",
                    "No verified hosted TARGET positives in training (only Helsinki). Hosted-target acceptance is measured via the 8 oracle manual boxes (never trained on).",
                    "Hosted negatives are DOTA-detected boats/harbor/vehicles (verifiable non-targets), excluding diagnostic frames 60/125."]}
json.dump(prov, open("/workspace/assets/bg_head_provenance.json", "w"), indent=2)
print("TRAIN:", Xtr.shape, "pos", int(ytr.sum()), "neg", int((ytr == 0).sum()))
print("HOLDOUT:", Xho.shape, "pos", int(yho.sum()), "neg", int((yho == 0).sum()))
print(json.dumps(prov, indent=2))
