"""V7 Stage A (final): soft-alpha matting for classes where binary segmentation failed.
alpha = normalised Mahalanobis distance to the surrounding background colour distribution,
modulated by a raised-cosine window over the GT box. Keeps thin structure (rotor blades,
lattice) at partial alpha and suppresses the original Helsinki background instead of pasting it."""
import json, glob, cv2, numpy as np
SRC = "/workspace/v6-diagnostic-release-8ae3b97/src/helsinki"; OUT = "/workspace/assets/v7_objects"
rep = json.load(open(f"{OUT}/extraction_report.json"))
SOFT = ["helicopter", "large_tower", "medium_launcher", "small_launcher", "small_tower"]
pick = {}
for f in sorted(glob.glob(f"{SRC}/annotations/*.json")):
    d = json.load(open(f))
    for a in d["annotations"]:
        k = a["object_id"]
        if k not in SOFT: continue
        x0, y0, x1, y1 = a["bbox"]
        if x0 < 50 or y0 < 50 or x1 > 3790 or y1 > 2110: continue
        ar = (x1-x0)*(y1-y0)
        if k not in pick or ar > pick[k][0]: pick[k] = (ar, d["frame"], [x0, y0, x1, y1])
tiles = []
for cls in SOFT:
    _, frame, box = pick[cls]
    img = cv2.imread(f"{SRC}/images/frame_{frame:06d}.png"); x0, y0, x1, y1 = box
    m = 10; P = img[y0-m:y1+m, x0-m:x1+m].astype(np.float32); h, w = P.shape[:2]
    ring = np.ones((h, w), bool); ring[m:h-m, m:w-m] = False
    px = P[ring].reshape(-1, 3); mu = px.mean(0); C = np.cov(px.T)+np.eye(3)*4.; Ci = np.linalg.inv(C)
    D = P.reshape(-1, 3)-mu; md = np.sqrt(np.einsum('ij,jk,ik->i', D, Ci, D)).reshape(h, w)
    lo, hi = np.percentile(md[ring], 75), max(np.percentile(md[ring], 99.5), 4.0)
    a = np.clip((md-lo)/max(1e-3, hi-lo), 0, 1)**0.7
    wy = np.hanning(h+2)[1:-1]**0.35; wx = np.hanning(w+2)[1:-1]**0.35   # soft box window, no hard edge
    a = a*np.outer(wy, wx)
    a = cv2.GaussianBlur(a.astype(np.float32), (3, 3), 0)
    a = np.clip(a/max(0.35, a.max()), 0, 1)
    np.savez_compressed(f"{OUT}/{cls}.npz", rgb=P.astype(np.uint8), alpha=a)
    rep[cls] = dict(frame=frame, gt_box=box, mask_mode="soft_alpha_matte", mean_alpha=round(float(a.mean()), 3),
                    cut_wh=[w, h], gt_wh=[x1-x0, y1-y0])
    chk = np.full_like(P, (255, 0, 255), np.uint8)
    comp = (P*a[..., None] + chk*(1-a[..., None])).astype(np.uint8)
    grn = np.full_like(P, (40, 90, 40), np.uint8)
    comp2 = (P*a[..., None] + grn*(1-a[..., None])).astype(np.uint8)
    t = np.hstack([cv2.resize(x, (180, 180)) for x in [P.astype(np.uint8), comp, comp2]])
    cv2.putText(t, f"{cls[:15]} mean_a {a.mean():.2f}", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1); tiles.append(t)
    print(cls, "soft matte f", frame, "mean_alpha", round(float(a.mean()), 3), [w, h])
json.dump(rep, open(f"{OUT}/extraction_report.json", "w"), indent=1)
cv2.imwrite(f"{OUT}/mask_soft.png", np.vstack(tiles))
print(json.dumps({k: v["mask_mode"] for k, v in rep.items()}))
