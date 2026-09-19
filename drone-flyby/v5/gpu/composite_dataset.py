"""Domain-adaptation dataset: feather-composite Helsinki target box-crops onto
varied backgrounds at diverse scale/rotation/position, plus background hard
negatives. One-class ('target'). Genuine views are kept; synthetic images are
clearly named. Boxes only (no masks) -> feathered alpha to soften rectangular seams.
Labels are approximate augmentation supervision, never claimed as new instances."""
import glob, os, math, shutil, json
import numpy as np, cv2

SRC = "/workspace/assets/dataset_v52"
OUT = "/workspace/assets/dataset_da"
rng = np.random.default_rng(20260919)
N_SYNTH = 600
N_NEG = 150
W, H = 960, 540


def load_targets_and_bgs():
    crops, bgs = [], []
    for img_path in sorted(glob.glob(SRC + "/images/train/*.png")):
        lab = SRC + "/labels/train/" + os.path.basename(img_path)[:-4] + ".txt"
        im = cv2.imread(img_path)
        if im is None:
            continue
        lines = [l for l in open(lab).read().splitlines() if l.strip()]
        boxes = []
        for l in lines:
            _, cx, cy, w, h = map(float, l.split())
            x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                boxes.append((x1, y1, x2, y2))
                crops.append(im[y1:y2, x1:x2].copy())
        # background: whole negative views, or images with few targets -> sample bg tiles avoiding boxes
        if not lines:
            bgs.append(im.copy())
        else:
            for _ in range(2):
                for _try in range(20):
                    bw = int(rng.integers(240, 700)); bh = int(bw * 9 // 16)
                    x = int(rng.integers(0, max(1, W - bw))); y = int(rng.integers(0, max(1, H - bh)))
                    g = (x, y, x + bw, y + bh)
                    if not any(min(g[2], b[2]) > max(g[0], b[0]) and min(g[3], b[3]) > max(g[1], b[1]) for b in boxes):
                        bgs.append(cv2.resize(im[y:y+bh, x:x+bw], (W, H)))
                        break
    return crops, bgs


def feather_alpha(h, w):
    a = np.ones((h, w), np.float32)
    m = max(1, int(min(h, w) * 0.18))
    ramp = np.linspace(0, 1, m)
    a[:m, :] *= ramp[:, None]; a[-m:, :] *= ramp[::-1, None]
    a[:, :m] *= ramp[None, :]; a[:, -m:] *= ramp[None, ::-1]
    return a


def paste(canvas, crop, cx, cy, target_w, rot_deg):
    ch, cw = crop.shape[:2]
    scale = target_w / max(1, cw)
    nw, nh = max(4, int(cw * scale)), max(4, int(ch * scale))
    c = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    a = feather_alpha(nh, nw)
    # rotate crop+alpha
    M = cv2.getRotationMatrix2D((nw/2, nh/2), rot_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    bw, bh = int(nh*sin + nw*cos), int(nh*cos + nw*sin)
    M[0, 2] += bw/2 - nw/2; M[1, 2] += bh/2 - nh/2
    c = cv2.warpAffine(c, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0)
    a = cv2.warpAffine(a, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0)
    x1, y1 = int(cx - bw/2), int(cy - bh/2)
    x2, y2 = x1 + bw, y1 + bh
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None
    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    am = a[:, :, None]
    canvas[y1:y2, x1:x2] = (c.astype(np.float32) * am + roi * (1 - am)).astype(np.uint8)
    # tight bbox from alpha
    ys, xs = np.where(a > 0.25)
    if len(xs) == 0:
        return None
    return (x1 + xs.min(), y1 + ys.min(), x1 + xs.max(), y1 + ys.max())


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for s in ("train", "calibration", "test"):
        os.makedirs(f"{OUT}/images/{s}", exist_ok=True)
        os.makedirs(f"{OUT}/labels/{s}", exist_ok=True)
    # keep genuine data
    for s in ("train", "calibration", "test"):
        for p in glob.glob(f"{SRC}/images/{s}/*.png"):
            shutil.copy2(p, f"{OUT}/images/{s}/")
        for p in glob.glob(f"{SRC}/labels/{s}/*.txt"):
            shutil.copy2(p, f"{OUT}/labels/{s}/")
    crops, bgs = load_targets_and_bgs()
    print(f"crop pool={len(crops)} bg pool={len(bgs)}")
    made = 0
    for i in range(N_SYNTH):
        bg = bgs[int(rng.integers(0, len(bgs)))].copy()
        if rng.random() < 0.3:  # photometric jitter on bg
            bg = np.clip(bg.astype(np.float32) * rng.uniform(0.7, 1.25) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
        k = int(rng.integers(2, 7))
        labels = []
        for _ in range(k):
            crop = crops[int(rng.integers(0, len(crops)))]
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
            tw = float(rng.choice([12, 16, 22, 30, 40, 55, 75], p=[.12, .18, .2, .2, .15, .1, .05]))
            cx = float(rng.uniform(tw, W - tw)); cy = float(rng.uniform(tw, H - tw))
            rot = float(rng.uniform(0, 360))
            b = paste(bg, crop, cx, cy, tw, rot)
            if b:
                labels.append(b)
        name = f"synth_{i:04d}"
        cv2.imwrite(f"{OUT}/images/train/{name}.png", bg)
        with open(f"{OUT}/labels/train/{name}.txt", "w") as f:
            for (x1, y1, x2, y2) in labels:
                f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        made += 1
    for i in range(N_NEG):
        bg = bgs[int(rng.integers(0, len(bgs)))].copy()
        name = f"hardneg_{i:04d}"
        cv2.imwrite(f"{OUT}/images/train/{name}.png", bg)
        open(f"{OUT}/labels/train/{name}.txt", "w").close()
    open(f"{OUT}/dataset.yaml", "w").write(
        f"path: {OUT}\ntrain: images/train\nval: images/calibration\ntest: images/test\nnames:\n  0: target\n")
    n_train = len(glob.glob(f"{OUT}/images/train/*.png"))
    json.dump({"genuine_train": len(glob.glob(f'{SRC}/images/train/*.png')), "synthetic": N_SYNTH,
               "hard_neg": N_NEG, "total_train": n_train, "crop_pool": len(crops), "bg_pool": len(bgs),
               "note": "Synthetic images are feathered box-crop composites for domain adaptation; approximate augmentation labels, not new instances."},
              open(f"{OUT}/da_provenance.json", "w"), indent=2)
    print(f"DONE total_train={n_train} (genuine+{made} synth+{N_NEG} neg)")


if __name__ == "__main__":
    main()
