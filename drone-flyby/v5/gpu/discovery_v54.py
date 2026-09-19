"""V5.4 merged discovery: DOTA yolo11l-obb (primary) + yolov8l-worldv2 (open-vocab
complementary). OBB->AABB via rotated corners, mapped to global coords. Merge/dedup
preserving neighboring objects; scale-aware spatially-diverse selection. Returns the
same candidate schema as v5.discovery.Discovery so it drops into v5.pipeline.Pipeline.

Also a RejectingExpert: adds reference-similarity-based background suppression to the
existing DINOv2 target head (no retraining) so vegetation/rooftops/boats are less
likely to be forced into an official class."""
import os, time
import numpy as np
from ultralytics import YOLO, YOLOWorld
from v2.geometry import iou, view_to_source
from v5.visual_expert import VisualExpert

WORLD_PROMPTS = ["airplane", "aircraft", "helicopter", "hangar", "military vehicle",
                 "tank", "truck", "tower", "radar dish", "missile launcher", "boat", "ship", "building"]


def _aabb_from_obb(poly):
    xs, ys = poly[:, 0], poly[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _tiles(W, H, cols, rows, ov):
    tw, th = W / cols, H / rows
    ox, oy = tw * ov, th * ov
    for r in range(rows):
        for c in range(cols):
            yield (int(max(0, c*tw-ox)), int(max(0, r*th-oy)), int(min(W, (c+1)*tw+ox)), int(min(H, (r+1)*th+oy)))


class MergedDiscovery:
    def __init__(self, assets=None, budget=None, device="cuda:0", threads=2,
                 obb_weights=None, world_weights=None, tiles_cols=0, tiles_rows=0):
        assets = assets or os.getenv("V5_ASSETS", "/workspace/assets")
        self.budget = int(budget or os.getenv("V5_CANDIDATES", "64"))
        self.obb_imgsz = int(os.getenv("V5_OBB_IMGSZ", "1280"))
        self.world_imgsz = int(os.getenv("V5_WORLD_IMGSZ", "1024"))
        self.obb_conf = float(os.getenv("V5_OBB_CONF", "0.10"))
        self.world_conf = float(os.getenv("V5_WORLD_CONF", "0.02"))
        self.tiles_cols = int(os.getenv("V5_TILES_COLS", str(tiles_cols)))
        self.tiles_rows = int(os.getenv("V5_TILES_ROWS", str(tiles_rows)))
        dev = 0 if device.startswith("cuda") else "cpu"
        self.dev = dev
        self.obb = YOLO(obb_weights or f"{assets}/yolo11l-obb.pt")
        self.world = YOLOWorld(world_weights or f"{assets}/yolov8l-worldv2.pt")
        self.world.set_classes(WORLD_PROMPTS)
        self.size = self.obb_imgsz
        # warmup
        z = np.zeros((540, 960, 3), np.uint8)
        self._obb(z); self._world(z)

    def _obb(self, img):
        r = self.obb.predict(img, conf=self.obb_conf, imgsz=self.obb_imgsz, verbose=False, device=self.dev)[0]
        out = []
        if r.obb is not None and len(r.obb) > 0:
            cs = r.obb.xyxyxyxy.cpu().numpy(); cf = r.obb.conf.cpu().numpy(); cl = r.obb.cls.cpu().numpy()
            names = r.names
            for poly, f, c in zip(cs, cf, cl):
                out.append((_aabb_from_obb(poly), float(f), f"obb:{names[int(c)]}"))
        return out

    def _world(self, img):
        r = self.world.predict(img, conf=self.world_conf, imgsz=self.world_imgsz, verbose=False, device=self.dev)[0]
        out = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b, f, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                out.append(([float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(f), f"world:{WORLD_PROMPTS[int(c)]}"))
        return out

    def _collect(self, image):
        H, W = image.shape[:2]
        raw = []
        for b, f, s in self._obb(image):
            raw.append((b, f, s))
        for b, f, s in self._world(image):
            raw.append((b, f, s))
        if self.tiles_cols and self.tiles_rows:
            for (x0, y0, x1, y1) in _tiles(W, H, self.tiles_cols, self.tiles_rows, 0.25):
                t = image[y0:y1, x0:x1]
                if t.size == 0:
                    continue
                for b, f, s in self._obb(t) + self._world(t):
                    raw.append(([b[0]+x0, b[1]+y0, b[2]+x0, b[3]+y0], f, s + "|tile"))
        # clip to frame
        clean = []
        for b, f, s in raw:
            bx = [max(0., b[0]), max(0., b[1]), min(float(W), b[2]), min(float(H), b[3])]
            if bx[2] - bx[0] >= 3 and bx[3] - bx[1] >= 3:
                clean.append((bx, f, s))
        return clean

    def _nms(self, dets, thr=0.6):
        dets = sorted(dets, key=lambda d: d[1], reverse=True)
        kept = []
        for b, f, s in dets:
            if all(iou(b, kb[0]) <= thr for kb in kept):
                kept.append((b, f, s))
        return kept

    def _diverse_select(self, merged, budget):
        """Scale-aware, spatially diverse over indices into `merged`: bucket by size and
        grid cell, round-robin highest-conf per (cell,scalebucket) until budget."""
        def bucket(b):
            area = (b[2]-b[0]) * (b[3]-b[1])
            sb = 0 if area < 900 else (1 if area < 6400 else 2)
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
            return (int(cx//160), int(cy//135), sb)
        groups = {}
        for i, (b, f, s) in enumerate(merged):
            groups.setdefault(bucket(b), []).append(i)  # merged already conf-sorted
        keys = list(groups.keys())
        order = []
        while len(order) < budget and any(groups[k] for k in keys):
            for k in keys:
                if groups[k]:
                    order.append(groups[k].pop(0))
                    if len(order) >= budget:
                        break
        return order  # list of indices into merged, in selection order

    def propose(self, image, region):
        start = time.perf_counter()
        dets = self._collect(image)
        collected = time.perf_counter()
        merged = self._nms(dets, thr=0.6)
        merged.sort(key=lambda d: d[1], reverse=True)
        sel_order = self._diverse_select(merged, self.budget)
        rank_of = {idx: r for r, idx in enumerate(sel_order, 1)}
        raw = []
        for i, (b, f, s) in enumerate(merged):
            src_box = list(view_to_source(b, region))
            crank = rank_of.get(i)
            sel = crank is not None and crank <= self.budget
            raw.append({"local_box": b, "discovery_score": float(f), "candidate_source": s,
                        "raw_rank": i + 1, "source_box": src_box,
                        "global_box": [v/sc for v, sc in zip(src_box, (3840, 2160, 3840, 2160))],
                        "candidate_rank": crank, "selected": sel,
                        "filtered_reason": (None if sel else ("recognition_budget" if crank else "nms")),
                        "emitted": False, "class_scores": None, "top3": None,
                        "target_probability": None, "margin": None, "entropy": None, "state": "not_classified"})
        sel_list = [r for r in raw if r["selected"]]
        stages = {"discovery_inference_ms": (collected-start)*1000,
                  "discovery_ranking_ms": (time.perf_counter()-collected)*1000,
                  "raw_count": len(raw), "nms_count": len(merged), "selected_count": len(sel_list)}
        return raw, sel_list, stages


class RejectingExpert(VisualExpert):
    """Background rejection on the frozen DINOv2 features.
    Primary: a supervised learned target/background head (bg_head.npz) whose P(target)
    replaces the weak sigmoid target head when V5_BG_HEAD=1 (default on if the file exists).
    Fallback: optional reference-similarity gate (V5_REF_TAU>0). 16-class head unchanged."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ref_tau = float(os.getenv("V5_REF_TAU", "0.0"))
        self.ref_temp = float(os.getenv("V5_REF_TEMP", "0.05"))
        self.bg = None
        path = os.path.join(str(self.assets), "bg_head.npz")
        if os.getenv("V5_BG_HEAD", "1") == "1" and os.path.exists(path):
            self.bg = dict(np.load(path, allow_pickle=False))
            self.bg["kind"] = str(self.bg["kind"])

    def _bg_prob(self, feats):
        b = self.bg
        if b["kind"] == "logreg":
            z = feats @ b["w"] + b["b"]
        else:  # mlp, single hidden relu
            h = np.maximum(0.0, feats @ b["W1"] + b["b1"])
            z = (h @ b["W2"] + b["b2"])[:, 0]
        return 1.0 / (1.0 + np.exp(-z))

    def classify_features(self, features):
        results = super().classify_features(features)
        if self.bg is not None and len(features):
            p = self._bg_prob(np.asarray(features, dtype=np.float32))
            for r, pv in zip(results, p):
                r["target_probability_orig"] = r["target_probability"]
                r["target_probability"] = float(pv)  # learned head is the gate
                r["state"] = "unknown" if pv < 0.5 else r["state"]
            return results
        if self.ref_tau > 0:
            for r in results:
                ref_max = max(r["reference_similarity"]) if r["reference_similarity"] else 0.0
                penalty = 1.0 / (1.0 + np.exp(-(ref_max - self.ref_tau) / self.ref_temp))
                r["target_probability"] = float(r["target_probability"] * penalty)
                r["state"] = "unknown" if r["target_probability"] < 0.5 else r["state"]
        return results
