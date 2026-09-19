"""Oracle-view L1/L2 feasibility: render genuine L1/L2 views centered on each Helsinki
GT object; measure whether the existing detector ensemble proposes it (source-IoU>=0.5),
survives selection, and what DINOv2 predicts. Minutes, not the whole task."""
import os, json, cv2, numpy as np
from v5.pipeline import Config
from v5.gpu.discovery_v54 import MergedDiscovery, RejectingExpert
from v5.discovery import Discovery as OldDiscovery
from v2.geometry import iou
from utils import load_frame, load_annotations, center_bounds_for_level, source_region_for_view
from dtos import OBJECT_CLASSES

cfg = Config()
merged = MergedDiscovery(cfg.assets, budget=64, device=cfg.device)
expert = RejectingExpert(cfg.assets, device=cfg.device)
FRAMES = [0, 6, 12, 18, 24]
LEVELS = [0, 1, 2]


def render_array(img, level, cx, cy):
    x1, y1, x2, y2 = source_region_for_view(level, cx, cy)
    v = img[y1:y2, x1:x2]
    if (v.shape[1], v.shape[0]) != (960, 540):
        v = cv2.resize(v, (960, 540), interpolation=cv2.INTER_AREA)
    return v, (x1, y1, x2, y2)


def clampc(level, cx, cy):
    mnx, mxx, mny, myy = center_bounds_for_level(level)
    return int(min(max(cx, mnx), mxx)), int(min(max(cy, mny), myy))


rows = {l: {"n": 0, "hit50": 0, "sel50": 0, "class_ok": 0, "ious": []} for l in LEVELS}
for frame in FRAMES:
    img = load_frame(frame, "helsinki")
    anns = load_annotations(frame, "helsinki")
    for level in LEVELS:
        for ann in anns:
            gb = ann["bbox"]
            cx, cy = (gb[0]+gb[2])//2, (gb[1]+gb[3])//2
            if level == 0:
                cx, cy = 1920, 1080
            else:
                cx, cy = clampc(level, cx, cy)
            view, region = render_array(img, level, cx, cy)
            raw, sel, _ = merged.propose(view, list(region))
            best_raw = max((iou(gb, c["source_box"]) for c in raw), default=0.0)
            best_sel = max((iou(gb, c["source_box"]) for c in sel), default=0.0)
            rows[level]["n"] += 1
            rows[level]["ious"].append(best_raw)
            if best_raw >= 0.5:
                rows[level]["hit50"] += 1
            if best_sel >= 0.5:
                rows[level]["sel50"] += 1
            # class check on best selected proposal (if any overlaps)
            cand = max(sel, key=lambda c: iou(gb, c["source_box"]), default=None) if sel else None
            if cand and iou(gb, cand["source_box"]) >= 0.3:
                res, _, _ = expert.classify(view, [cand["local_box"]])
                if res[0]["top3"][0]["class"] == ann["object_id"]:
                    rows[level]["class_ok"] += 1
print("level | n | raw_recall@0.5 | selected_recall@0.5 | mean_best_iou | class_ok(top1 on proposal)")
for l in LEVELS:
    r = rows[l]
    print(f"L{l} | {r['n']} | {r['hit50']/r['n']:.2f} | {r['sel50']/r['n']:.2f} | {np.mean(r['ious']):.3f} | {r['class_ok']}/{r['n']}")
json.dump(rows, open("/workspace/results/feasibility_l1l2.json", "w"), default=float, indent=2)
print("WROTE /workspace/results/feasibility_l1l2.json")
