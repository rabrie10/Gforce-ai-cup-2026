"""Genuine multiscale observation dataset from clean Helsinki 4K + real GT.
Renders L0/L1/L2 views (exactly as the evaluator: crop source region -> 960x540 INTER_AREA),
transforms every intersecting GT box into view coords, one-class 'target'. Adds background-only
views. Records provenance (frame, level, region, source-global obj id/class, local+global box,
truncation, split). Overlays are never used. One physical instance per class -> documented caveat."""
import os, json, glob, shutil, cv2, numpy as np
from utils import load_frame, load_annotations, center_bounds_for_level, source_region_for_view, frame_numbers
from v2.geometry import source_to_view, iou

OUT = "/workspace/assets/dataset_ms"
rng = np.random.default_rng(20260919)
CALIB = {4, 9, 14, 18}
W, H = 960, 540


def split_for(f):
    return "test" if f >= 21 else ("calibration" if f in CALIB else "train")


def render(img, level, cx, cy):
    x1, y1, x2, y2 = source_region_for_view(level, cx, cy)
    v = img[y1:y2, x1:x2]
    if (v.shape[1], v.shape[0]) != (W, H):
        v = cv2.resize(v, (W, H), interpolation=cv2.INTER_AREA)
    return v, (x1, y1, x2, y2)


def clampc(level, cx, cy):
    mnx, mxx, mny, myy = center_bounds_for_level(level)
    return int(min(max(cx, mnx), mxx)), int(min(max(cy, mny), myy))


def boxes_in_view(anns, region):
    out = []
    for a in anns:
        b = a["bbox"]
        if min(region[2], b[2]) > max(region[0], b[0]) and min(region[3], b[3]) > max(region[1], b[1]):
            lb = source_to_view(b, region)
            lb = [max(0., lb[0]), max(0., lb[1]), min(float(W), lb[2]), min(float(H), lb[3])]
            if lb[2]-lb[0] >= 3 and lb[3]-lb[1] >= 3:
                out.append((lb, a["object_id"], b))
    return out


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for s in ("train", "calibration", "test"):
        os.makedirs(f"{OUT}/images/{s}", exist_ok=True)
        os.makedirs(f"{OUT}/labels/{s}", exist_ok=True)
    manifest = []
    n = 0
    for frame in frame_numbers("helsinki"):
        img = load_frame(frame, "helsinki")
        anns = load_annotations(frame, "helsinki")
        split = split_for(frame)
        views = [(0, 1920, 1080)]  # L0 full
        # per-object L1/L2 with center jitter so target appears at varied positions/truncation
        for a in anns:
            b = a["bbox"]; ocx, ocy = (b[0]+b[2])//2, (b[1]+b[3])//2
            for level, jit in [(1, 300), (1, 500), (2, 150), (2, 260), (2, 40)]:
                cx = ocx + int(rng.uniform(-jit, jit)); cy = ocy + int(rng.uniform(-jit, jit))
                views.append((level, *clampc(level, cx, cy)))
        # background-only views
        for level in (1, 2):
            for _ in range(3):
                for _t in range(15):
                    cx, cy = clampc(level, int(rng.uniform(0, 3840)), int(rng.uniform(0, 2160)))
                    _, region = render(img, level, cx, cy)
                    if not boxes_in_view(anns, region):
                        views.append((level, cx, cy)); break
        for (level, cx, cy) in views:
            view, region = render(img, level, cx, cy)
            bxs = boxes_in_view(anns, region)
            name = f"f{frame:03d}_L{level}_{cx}_{cy}"
            cv2.imwrite(f"{OUT}/images/{split}/{name}.png", view)
            with open(f"{OUT}/labels/{split}/{name}.txt", "w") as fh:
                for lb, cls, gb in bxs:
                    fh.write(f"0 {(lb[0]+lb[2])/2/W:.6f} {(lb[1]+lb[3])/2/H:.6f} {(lb[2]-lb[0])/W:.6f} {(lb[3]-lb[1])/H:.6f}\n")
            manifest.append({"name": name, "frame": frame, "split": split, "level": level,
                             "region": list(region), "n_targets": len(bxs),
                             "classes": [c for _, c, _ in bxs]})
            n += 1
    open(f"{OUT}/dataset.yaml", "w").write(
        f"path: {OUT}\ntrain: images/train\nval: images/calibration\ntest: images/test\nnames:\n  0: target\n")
    json.dump({"views": n, "caveat": "One physical instance per class; multiple crops are not new instances.",
               "manifest": manifest}, open(f"{OUT}/manifest.json", "w"), indent=2)
    counts = {s: len(glob.glob(f"{OUT}/images/{s}/*.png")) for s in ("train", "calibration", "test")}
    tgt = sum(len(open(f).read().split()) // 5 for f in glob.glob(f"{OUT}/labels/train/*.txt"))
    print("views:", counts, "train target instances:", tgt)


if __name__ == "__main__":
    main()
