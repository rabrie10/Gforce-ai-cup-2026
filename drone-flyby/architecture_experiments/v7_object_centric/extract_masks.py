"""V7 Stage A: extract per-class foreground masks from Helsinki 4K frames using GrabCut
seeded by the official GT boxes (training assets only; no hosted data).
Picks, per class, the largest un-clipped instance across the 25 frames.
Quality gates reject degenerate masks -> feathered-box fallback (recorded)."""
import json, glob, os, cv2, numpy as np

SRC = "/workspace/v6-diagnostic-release-8ae3b97/src/helsinki"
OUT = "/workspace/assets/v7_objects"
SEED = 20260920
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

best = {}
for f in sorted(glob.glob(f"{SRC}/annotations/*.json")):
    d = json.load(open(f))
    for a in d["annotations"]:
        x0, y0, x1, y1 = a["bbox"]
        if x0 < 40 or y0 < 40 or x1 > 3800 or y1 > 2120:   # need margin for context
            continue
        area = (x1-x0)*(y1-y0)
        k = a["object_id"]
        if k not in best or area > best[k][0]:
            best[k] = (area, d["frame"], [x0, y0, x1, y1])

report = {}
tiles = []
for cls, (area, frame, box) in sorted(best.items()):
    img = cv2.imread(f"{SRC}/images/frame_{frame:06d}.png")
    x0, y0, x1, y1 = box
    m = 24                                  # context margin fed to GrabCut
    X0, Y0, X1, Y1 = x0-m, y0-m, x1+m, y1+m
    patch = img[Y0:Y1, X0:X1].copy()
    gc = np.zeros(patch.shape[:2], np.uint8)
    gc[:] = cv2.GC_BGD
    gc[m-6:patch.shape[0]-m+6, m-6:patch.shape[1]-m+6] = cv2.GC_PR_BGD   # box+slack = probable fg region
    gc[m+2:patch.shape[0]-m-2, m+2:patch.shape[1]-m-2] = cv2.GC_PR_FGD
    ch, cw = patch.shape[0]//2, patch.shape[1]//2
    gc[ch-2:ch+3, cw-2:cw+3] = cv2.GC_FGD                                 # centre seed
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(patch, gc, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
        mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception as e:
        mask = np.zeros(patch.shape[:2], np.uint8); print(cls, "grabcut failed", e)
    mask[:m-6, :] = 0; mask[-(m-6):, :] = 0; mask[:, :m-6] = 0; mask[:, -(m-6):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n > 1:   # keep components that are large or touch the object centre region
        keep = np.zeros_like(mask)
        big = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        for i in range(1, n):
            if i == big or stats[i, cv2.CC_STAT_AREA] > 0.04*stats[big, cv2.CC_STAT_AREA]:
                keep[lab == i] = 255
        mask = keep
    boxarea = (x1-x0)*(y1-y0)
    ratio = float((mask > 0).sum()) / boxarea
    ok = 0.08 <= ratio <= 0.98
    ys, xs = np.where(mask > 0)
    if ok and len(xs):
        # tight extent of the extracted silhouette, in patch coords
        ex = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
        cover = ((ex[2]-ex[0])*(ex[3]-ex[1])) / boxarea
        ok = cover >= 0.35        # silhouette must span a real part of the GT box
    if not ok:
        mask = np.zeros(patch.shape[:2], np.uint8)
        mask[m:patch.shape[0]-m, m:patch.shape[1]-m] = 255
        ex = [m, m, patch.shape[1]-m, patch.shape[0]-m]
    alpha = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)/255.0
    cut = patch[ex[1]:ex[3], ex[0]:ex[2]]
    cuta = alpha[ex[1]:ex[3], ex[0]:ex[2]]
    np.savez_compressed(f"{OUT}/{cls}.npz", rgb=cut, alpha=cuta)
    report[cls] = dict(frame=frame, gt_box=box, mask_mode="grabcut" if ok else "FALLBACK_box",
                       fg_ratio=round(ratio, 3), cut_wh=[int(cut.shape[1]), int(cut.shape[0])],
                       gt_wh=[x1-x0, y1-y0])
    # contact-sheet tile: original | mask overlay | cutout on magenta
    vis = patch.copy(); ov = vis.copy(); ov[mask > 0] = (0, 255, 0)
    ov = cv2.addWeighted(vis, .55, ov, .45, 0)
    chk = np.full_like(cut, (255, 0, 255), np.uint8)
    comp = (cut*cuta[..., None] + chk*(1-cuta[..., None])).astype(np.uint8)
    row = [vis, ov, cv2.copyMakeBorder(comp, 0, max(0, vis.shape[0]-comp.shape[0]), 0, max(0, vis.shape[1]-comp.shape[1]), cv2.BORDER_CONSTANT)]
    row = [cv2.resize(r, (150, 150)) for r in row]
    t = np.hstack(row)
    cv2.putText(t, f"{cls[:14]} {report[cls]['mask_mode'][:8]} {ratio:.2f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
    tiles.append(t)
json.dump(report, open(f"{OUT}/extraction_report.json", "w"), indent=1)
rows = [np.hstack(tiles[i:i+4]) for i in range(0, 16, 4)]
cv2.imwrite(f"{OUT}/mask_contact_sheet.png", np.vstack(rows))
for k, v in report.items(): print(k, v["mask_mode"], v["fg_ratio"], v["cut_wh"], v["gt_wh"])
