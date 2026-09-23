"""V7 object-centric domain-randomised multiscale dataset.

Superset of the ms1 distribution: every image is a GENUINE evaluator-style observation
(source region -> 960x540 INTER_AREA, exactly as build_multiscale/the evaluator), with all
intersecting official Helsinki GT boxes kept as positives, PLUS extra masked object cut-outs
pasted with randomised orientation / scale / position / appearance / clipping / occlusion.

Training assets only (Helsinki 4K frames + official Helsinki annotations + masks extracted from
them). No hosted diagnostic image or annotation is read by this script.

Box convention: pasted boxes are the tight alpha extent expanded by a per-class factor measured
against that class's official Helsinki GT box, so pasted and genuine labels share one convention.
The hidden hosted annotation convention may still differ.
"""
import os, sys, json, glob, math, argparse, hashlib
import numpy as np, cv2

sys.path.insert(0, "/workspace/v6-diagnostic-release-8ae3b97")
from utils import source_region_for_view, center_bounds_for_level  # evaluator geometry

SRC = "/workspace/v6-diagnostic-release-8ae3b97/src/helsinki"
OBJ = "/workspace/assets/v7_objects"
W, H = 960, 540
CALIB = {4, 9, 14, 18}          # same split convention as dataset_ms
ALPHA_T = 0.35                  # alpha level defining the object's visible extent


def split_for(f):
    return "test" if f >= 21 else ("val" if f in CALIB else "train")


def load_objects():
    rep = json.load(open(f"{OBJ}/extraction_report.json"))
    objs = {}
    for cls, meta in rep.items():
        d = np.load(f"{OBJ}/{cls}.npz")
        rgb, a = d["rgb"], d["alpha"].astype(np.float32)
        if meta["mask_mode"].startswith("soft"):
            # soft mattes are mostly transparent -> stretch so the object body becomes opaque
            a = np.clip((a - 0.10)/0.45, 0, 1)
        ys, xs = np.where(a >= ALPHA_T)
        if not len(xs):
            continue
        tw, th = xs.max()-xs.min()+1, ys.max()-ys.min()+1
        gw, gh = meta["gt_wh"]
        # per-class convention factor: official GT box vs tight alpha extent (isotropic)
        fac = float(np.clip(math.sqrt((gw/max(1, tw))*(gh/max(1, th))), 0.85, 1.6))
        objs[cls] = dict(rgb=rgb, alpha=a, conv=fac, mode=meta["mask_mode"])
    return objs


def rot_scale(rgb, alpha, deg, target_long):
    """Rotate about the object's alpha centroid and scale so the visible long side ~= target_long."""
    ys, xs = np.where(alpha >= ALPHA_T)
    tw, th = xs.max()-xs.min()+1, ys.max()-ys.min()+1
    s = target_long / max(tw, th)
    h, w = alpha.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, s)
    nw, nh = int(math.ceil(w*abs(s)*1.6))+4, int(math.ceil(h*abs(s)*1.6))+4
    M[0, 2] += nw/2 - w/2; M[1, 2] += nh/2 - h/2
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.warpAffine(rgb, M, (nw, nh), flags=interp, borderValue=(0, 0, 0))
    a = cv2.warpAffine(alpha, M, (nw, nh), flags=interp, borderValue=0)
    return r, np.clip(a, 0, 1)


def jitter_object(rgb, rng, bg_mean):
    f = rgb.astype(np.float32)
    f *= rng.uniform(0.78, 1.22)                                  # exposure
    f = (f - f.mean())*rng.uniform(0.82, 1.18) + f.mean()          # contrast
    f += (bg_mean - f.reshape(-1, 3).mean(0))*rng.uniform(0.0, 0.35)   # partial colour adaptation
    hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.integers(-7, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1]*rng.uniform(0.75, 1.2), 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def paste(canvas, rgb, alpha, cx, cy, rng, shadow_dir, min_delta=8.0):
    """Alpha-composite with a soft drop shadow. Returns the visible tight box or None.
    A visibility gate reverts the paste (and drops the label) when the composite is not
    visibly different from the background - no invisible 'ghost' positives."""
    before = canvas.copy()
    h, w = alpha.shape
    x0, y0 = int(round(cx - w/2)), int(round(cy - h/2))
    # shadow first (physically plausible, consistent direction per image)
    sh_dx, sh_dy = shadow_dir
    sa = cv2.GaussianBlur(alpha, (0, 0), max(1.0, 0.04*max(w, h)))*rng.uniform(0.18, 0.45)
    for src_a, dx, dy, col in ((sa, sh_dx, sh_dy, 0.0), (alpha, 0, 0, None)):
        xs0, ys0 = x0+dx, y0+dy
        X0, Y0 = max(0, xs0), max(0, ys0)
        X1, Y1 = min(W, xs0+w), min(H, ys0+h)
        if X1 <= X0 or Y1 <= Y0:
            continue
        sx0, sy0 = X0-xs0, Y0-ys0
        a = src_a[sy0:sy0+(Y1-Y0), sx0:sx0+(X1-X0)][..., None]
        roi = canvas[Y0:Y1, X0:X1].astype(np.float32)
        src = (roi*0.35) if col is not None else rgb[sy0:sy0+(Y1-Y0), sx0:sx0+(X1-X0)].astype(np.float32)
        canvas[Y0:Y1, X0:X1] = np.clip(roi*(1-a) + src*a, 0, 255).astype(np.uint8)
    vis = np.zeros((H, W), np.float32)
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0+w), min(H, y0+h)
    if X1 <= X0 or Y1 <= Y0:
        return None, 0.0
    vis[Y0:Y1, X0:X1] = alpha[Y0-y0:Y1-y0, X0-x0:X1-x0]
    keep = vis.sum()/max(1e-6, alpha.sum())
    ys, xs = np.where(vis >= ALPHA_T)
    if not len(xs):
        canvas[:] = before; return None, keep
    bx = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
    d = np.abs(canvas[bx[1]:bx[3], bx[0]:bx[2]].astype(np.float32) - before[bx[1]:bx[3], bx[0]:bx[2]].astype(np.float32))
    if d.size == 0 or d.mean() < min_delta or d.max() < 25:
        canvas[:] = before; return None, keep      # invisible paste -> no object, no label
    return bx, float(keep)


def occlude(canvas, box, bgpatch, rng):
    """Overlay an elliptical patch of scene texture over part of the object."""
    x0, y0, x1, y1 = box
    cx = rng.integers(x0, max(x0+1, x1)); cy = rng.integers(y0, max(y0+1, y1))
    rx = max(3, int((x1-x0)*rng.uniform(0.2, 0.45))); ry = max(3, int((y1-y0)*rng.uniform(0.2, 0.45)))
    m = np.zeros((H, W), np.uint8); cv2.ellipse(m, (int(cx), int(cy)), (rx, ry), 0, 0, 360, 255, -1)
    m = cv2.GaussianBlur(m, (5, 5), 0).astype(np.float32)[..., None]/255.
    canvas[:] = np.clip(canvas*(1-m) + bgpatch*m, 0, 255).astype(np.uint8)


def global_appearance(img, rng):
    f = img.astype(np.float32)
    f = ((f/255.)**rng.uniform(0.82, 1.2))*255.                       # gamma
    f *= rng.uniform(0.85, 1.15)
    f = (f-f.mean())*rng.uniform(0.88, 1.15) + f.mean()
    if rng.random() < 0.35:                                            # haze
        f = f*(1-0.12) + rng.uniform(120, 200)*0.12
    img = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.9))
    if rng.random() < 0.35:
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(1.5, 5.0), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        q = int(rng.integers(55, 92))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/assets/dataset_v7")
    ap.add_argument("--n-train", type=int, default=3200)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--sheet-only", action="store_true")
    ap.add_argument("--tag", default="a")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    objs = load_objects()
    frames, anns = {}, {}
    for f in sorted(glob.glob(f"{SRC}/annotations/*.json")):
        d = json.load(open(f)); anns[d["frame"]] = d["annotations"]
    for n in sorted(anns):
        frames[n] = cv2.imread(f"{SRC}/images/frame_{n:06d}.png")
    tr_f = [n for n in frames if split_for(n) == "train"]
    va_f = [n for n in frames if split_for(n) == "val"]
    print("train frames", tr_f, "val frames", va_f, "objects", len(objs))

    def render(fn, level, cx, cy):
        x0, y0, x1, y1 = source_region_for_view(level, cx, cy)
        v = frames[fn][y0:y1, x0:x1]
        if (v.shape[1], v.shape[0]) != (W, H):
            v = cv2.resize(v, (W, H), interpolation=cv2.INTER_AREA)
        return v.copy(), (x0, y0, x1, y1)

    def build(fn, rng):
        level = int(rng.choice([0, 1, 2], p=[0.25, 0.40, 0.35]))
        mnx, mxx, mny, myy = center_bounds_for_level(level)
        cx = int(rng.integers(mnx, mxx+1)); cy = int(rng.integers(mny, myy+1))
        view, reg = render(fn, level, cx, cy)
        s = (reg[2]-reg[0])/W                      # source px per view px
        boxes = []
        for an in anns[fn]:                        # genuine targets, official convention
            gx0, gy0, gx1, gy1 = an["bbox"]
            bx = [(gx0-reg[0])/s, (gy0-reg[1])/s, (gx1-reg[0])/s, (gy1-reg[1])/s]
            ix0, iy0 = max(0., bx[0]), max(0., bx[1]); ix1, iy1 = min(float(W), bx[2]), min(float(H), bx[3])
            if ix1-ix0 < 3 or iy1-iy0 < 3:
                continue
            if (ix1-ix0)*(iy1-iy0) < 0.30*max(1e-6, (bx[2]-bx[0])*(bx[3]-bx[1])):
                continue                            # mostly outside the view
            boxes.append([ix0, iy0, ix1, iy1])
        bg_mean = view.reshape(-1, 3).mean(0)
        sdir = (int(rng.integers(-6, 7)), int(rng.integers(-6, 7)))
        npaste = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.10, 0.22, 0.24, 0.20, 0.14, 0.10]))
        bgcopy = view.copy()
        for _ in range(npaste):
            cls = str(rng.choice(list(objs)))
            o = objs[cls]
            # target size in OBSERVATION pixels: log-uniform, weighted to the small/medium regime
            tl = float(np.exp(rng.uniform(math.log(9), math.log(150))))
            deg = float(rng.uniform(0, 360))
            rgb, al = rot_scale(o["rgb"], o["alpha"], deg, tl)
            if rgb.shape[0] < 3 or rgb.shape[1] < 3:
                continue
            if rng.random() < 0.5:
                rgb, al = rgb[:, ::-1], al[:, ::-1]
            rgb = jitter_object(rgb, rng, bg_mean)
            if tl < 34:      # small objects lose detail through the evaluator's downsampling chain
                k = 0.45 if tl > 18 else 0.75
                rgb = cv2.GaussianBlur(rgb, (0, 0), k); al = cv2.GaussianBlur(al, (0, 0), k*0.7)
            clip = rng.random() < 0.18                    # deliberately border-clipped example
            if clip:
                side = int(rng.integers(0, 4))
                px, py = (rng.uniform(0, W), 0.0) if side == 0 else (rng.uniform(0, W), float(H)) if side == 1 \
                    else (0.0, rng.uniform(0, H)) if side == 2 else (float(W), rng.uniform(0, H))
                px += rng.uniform(-.25, .25)*rgb.shape[1]; py += rng.uniform(-.25, .25)*rgb.shape[0]
            else:
                px, py = rng.uniform(0.04*W, 0.96*W), rng.uniform(0.04*H, 0.96*H)
            box, keep = paste(view, rgb, al, px, py, rng, sdir)
            if box is None or keep < 0.35:
                continue
            bw, bh = box[2]-box[0], box[3]-box[1]
            if min(bw, bh) < 3 or max(bw, bh) < 6:
                continue
            if rng.random() < 0.12:
                occlude(view, box, bgcopy, rng)
            c = o["conv"]                                  # tight extent -> official-style convention
            ccx, ccy = (box[0]+box[2])/2, (box[1]+box[3])/2
            eb = [ccx-bw*c/2, ccy-bh*c/2, ccx+bw*c/2, ccy+bh*c/2]
            eb = [max(0., eb[0]), max(0., eb[1]), min(float(W), eb[2]), min(float(H), eb[3])]
            if eb[2]-eb[0] >= 3 and eb[3]-eb[1] >= 3:
                boxes.append(eb)
        view = global_appearance(view, rng)
        return view, boxes, dict(frame=fn, level=level, center=[cx, cy], region=list(reg), n_pasted=npaste)

    if a.sheet_only:
        tiles = []
        for i in range(12):
            v, b, meta = build(int(rng.choice(tr_f)), rng)
            for q in b:
                cv2.rectangle(v, (int(q[0]), int(q[1])), (int(q[2]), int(q[3])), (255, 0, 255), 1)
            cv2.putText(v, f"L{meta['level']} f{meta['frame']} n={len(b)}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            tiles.append(cv2.resize(v, (480, 270)))
        cv2.imwrite("/workspace/assets/dataset_v7_sheet.png", np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, 12, 3)]))
        print("sheet written"); return

    for sp in ("train", "val"):
        os.makedirs(f"{a.out}/images/{sp}", exist_ok=True); os.makedirs(f"{a.out}/labels/{sp}", exist_ok=True)
    prov, stats = [], dict(imgs=0, boxes=0, empty=0, bad=0)
    for sp, n, pool in (("train", a.n_train, tr_f), ("val", a.n_val, va_f)):
        for i in range(n):
            fn = int(rng.choice(pool))
            view, boxes, meta = build(fn, rng)
            assert view.shape == (H, W, 3)
            lines = []
            for b in boxes:
                x0, y0, x1, y1 = b
                if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H) or not np.isfinite([x0, y0, x1, y1]).all():
                    stats["bad"] += 1; continue
                cx, cy, bw, bh = (x0+x1)/2/W, (y0+y1)/2/H, (x1-x0)/W, (y1-y0)/H
                if bw <= 0 or bh <= 0 or cx <= 0 or cy <= 0 or cx >= 1 or cy >= 1:
                    stats["bad"] += 1; continue
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            name = f"{sp}_{a.tag}{i:05d}_f{meta['frame']:02d}_L{meta['level']}"
            cv2.imwrite(f"{a.out}/images/{sp}/{name}.jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(f"{a.out}/labels/{sp}/{name}.txt", "w").write("\n".join(lines))
            stats["imgs"] += 1; stats["boxes"] += len(lines); stats["empty"] += (len(lines) == 0)
            meta.update(name=name, split=sp, n_boxes=len(lines)); prov.append(meta)
    open(f"{a.out}/dataset.yaml", "w").write(
        f"path: {a.out}\ntrain: images/train\nval: images/val\nnames:\n  0: target\n")
    json.dump(dict(tag=a.tag, seed=a.seed, stats=stats, caveat="Pasted objects are transformed copies of the SAME ~one physical instance per class; they are not new physical instances.",
                   sources="Helsinki 4K frames + official Helsinki annotations + masks extracted from them. No hosted data.",
                   objects={k: v["mode"] for k, v in objs.items()}, conv={k: round(v["conv"], 3) for k, v in objs.items()},
                   provenance=prov[:200]), open(f"{a.out}/manifest_{a.tag}.json", "w"), indent=1)
    print("STATS", stats)


if __name__ == "__main__":
    main()
