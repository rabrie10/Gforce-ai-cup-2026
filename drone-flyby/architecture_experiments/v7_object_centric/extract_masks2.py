"""V7 Stage A (pass 2): for classes where GrabCut failed, segment by local background statistics
(Mahalanobis distance to the colour distribution of the ring OUTSIDE the GT box), which keeps thin
structures (rotor blades, lattice) that GrabCut drops. Tries several frames, keeps the best mask."""
import json, glob, os, cv2, numpy as np
SRC = "/workspace/v6-diagnostic-release-8ae3b97/src/helsinki"; OUT = "/workspace/assets/v7_objects"
rep = json.load(open(f"{OUT}/extraction_report.json"))
todo = [k for k, v in rep.items() if v["mask_mode"] != "grabcut"] + ["small_tower", "jammer"]
cands = {k: [] for k in todo}
for f in sorted(glob.glob(f"{SRC}/annotations/*.json")):
    d = json.load(open(f))
    for a in d["annotations"]:
        if a["object_id"] not in todo: continue
        x0, y0, x1, y1 = a["bbox"]
        if x0 < 50 or y0 < 50 or x1 > 3790 or y1 > 2110: continue
        cands[a["object_id"]].append((-(x1-x0)*(y1-y0), d["frame"], [x0, y0, x1, y1]))
tiles = []
for cls in todo:
    lst = sorted(cands[cls])[:4]
    bestm = None
    for _, frame, box in lst:
        img = cv2.imread(f"{SRC}/images/frame_{frame:06d}.png"); x0, y0, x1, y1 = box
        m = 26; P = img[y0-m:y1+m, x0-m:x1+m].astype(np.float32)
        h, w = P.shape[:2]
        ring = np.ones((h, w), bool); ring[m-4:h-m+4, m-4:w-m+4] = False
        px = P[ring].reshape(-1, 3)
        mu = px.mean(0); C = np.cov(px.T) + np.eye(3)*6.0; Ci = np.linalg.inv(C)
        D = P.reshape(-1, 3) - mu
        md = np.sqrt(np.einsum('ij,jk,ik->i', D, Ci, D)).reshape(h, w)
        inner = np.zeros((h, w), bool); inner[m-3:h-m+3, m-3:w-m+3] = True
        thr = max(3.0, np.percentile(md[ring], 99.0))
        mask = ((md > thr) & inner).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        if n > 1:
            big = max(range(1, n), key=lambda i: st[i, cv2.CC_STAT_AREA])
            keep = np.zeros_like(mask)
            for i in range(1, n):
                if i == big or st[i, cv2.CC_STAT_AREA] > 0.02*st[big, cv2.CC_STAT_AREA]: keep[lab == i] = 255
            mask = keep
        ys, xs = np.where(mask > 0)
        if not len(xs): continue
        ba = (x1-x0)*(y1-y0); ratio = (mask > 0).sum()/ba
        ex = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
        cover = ((ex[2]-ex[0])*(ex[3]-ex[1]))/ba
        score = cover - abs(ratio-0.35)*0.3
        if 0.05 <= ratio <= 0.9 and cover >= 0.45 and (bestm is None or score > bestm[0]):
            bestm = (score, frame, box, P.astype(np.uint8), mask, ex, ratio, cover)
    if bestm is None: print(cls, "still failing"); continue
    _, frame, box, P, mask, ex, ratio, cover = bestm
    alpha = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)/255.
    cut, cuta = P[ex[1]:ex[3], ex[0]:ex[2]], alpha[ex[1]:ex[3], ex[0]:ex[2]]
    np.savez_compressed(f"{OUT}/{cls}.npz", rgb=cut, alpha=cuta)
    rep[cls] = dict(frame=frame, gt_box=box, mask_mode="bgstats", fg_ratio=round(float(ratio), 3),
                    cover=round(float(cover), 3), cut_wh=[int(cut.shape[1]), int(cut.shape[0])], gt_wh=[box[2]-box[0], box[3]-box[1]])
    print(cls, "bgstats f", frame, "ratio", round(float(ratio), 2), "cover", round(float(cover), 2))
    ov = P.copy(); ov[mask > 0] = (0, 255, 0); ov = cv2.addWeighted(P, .5, ov, .5, 0)
    chk = np.full_like(cut, (255, 0, 255), np.uint8)
    comp = (cut*cuta[..., None] + chk*(1-cuta[..., None])).astype(np.uint8)
    t = np.hstack([cv2.resize(x, (170, 170)) for x in [P, ov, comp]])
    cv2.putText(t, f"{cls[:15]} {ratio:.2f}", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    tiles.append(t)
json.dump(rep, open(f"{OUT}/extraction_report.json", "w"), indent=1)
if tiles: cv2.imwrite(f"{OUT}/mask_pass2.png", np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles)-len(tiles) % 3, 3)] or [np.hstack(tiles)]))
print(json.dumps({k: v["mask_mode"] for k, v in rep.items()}))
