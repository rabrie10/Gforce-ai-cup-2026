"""QC: render pasted objects only (no genuine GT) and zoom on each, to confirm a human can see
the object inside every positive label. Catches near-invisible soft-matte ghosts."""
import sys, numpy as np, cv2
sys.argv = ["x"]; sys.path.insert(0, "/workspace")
import gen_v7_dataset as G
rng = np.random.default_rng(7)
objs = G.load_objects()
import json, glob
anns = {}; frames = {}
for f in sorted(glob.glob(f"{G.SRC}/annotations/*.json")):
    d = json.load(open(f)); anns[d["frame"]] = d["annotations"]
for n in (0, 6, 12, 20): frames[n] = cv2.imread(f"{G.SRC}/images/frame_{n:06d}.png")
tiles = []
for cls in sorted(objs):
    for tl in (14., 30., 70.):
        o = objs[cls]
        fn = int(rng.choice(list(frames)))
        x0, y0 = int(rng.integers(0, 2800)), int(rng.integers(0, 1500))
        view = cv2.resize(frames[fn][y0:y0+540, x0:x0+960], (960, 540), interpolation=cv2.INTER_AREA).copy()
        rgb, al = G.rot_scale(o["rgb"], o["alpha"], float(rng.uniform(0, 360)), tl)
        rgb = G.jitter_object(rgb, rng, view.reshape(-1, 3).mean(0))
        if tl < 34:
            k = .45 if tl > 18 else .75
            rgb = cv2.GaussianBlur(rgb, (0, 0), k); al = cv2.GaussianBlur(al, (0, 0), k*.7)
        box, keep = G.paste(view, rgb, al, 480, 270, rng, (4, 4))
        if box is None: print("MISS", cls, tl); continue
        p = 26; c = view[max(0, box[1]-p):box[3]+p, max(0, box[0]-p):box[2]+p]
        c = cv2.resize(c, None, fx=max(1, int(150/max(c.shape[:2]))), fy=max(1, int(150/max(c.shape[:2]))), interpolation=cv2.INTER_NEAREST)
        bb = [(box[0]-max(0, box[0]-p)), (box[1]-max(0, box[1]-p))]
        sc = c.shape[0]/max(1, (box[3]+p-max(0, box[1]-p)))
        cv2.rectangle(c, (int(bb[0]*sc), int(bb[1]*sc)), (int((bb[0]+box[2]-box[0])*sc), int((bb[1]+box[3]-box[1])*sc)), (255, 0, 255), 1)
        c = cv2.resize(c, (170, 170))
        cv2.putText(c, f"{cls[:12]} {int(tl)}px", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 255), 1)
        tiles.append(c)
rows = [np.hstack(tiles[i:i+6]) for i in range(0, len(tiles)-len(tiles) % 6, 6)]
cv2.imwrite("/workspace/assets/qc_paste.png", np.vstack(rows)); print("ok", len(tiles))
