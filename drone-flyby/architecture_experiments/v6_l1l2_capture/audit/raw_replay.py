"""Stage-1..3 replay on Runpod (run with PYTHONPATH=<deployed release>): exact ms1 checkpoint, exact production
preprocessing (cv2 BGR PNG -> YOLO.predict imgsz=960), only conf lowered. Does NOT touch port 9053.
Usage: python raw_replay.py <audit_dir>   (audit_dir has captures/*.png|json and manual_annotations_frozen.json)"""
import sys, json, glob, os, time
import numpy as np, cv2
from ultralytics import YOLO
audit = sys.argv[1]
W0 = "/workspace/training/ms1/weights/best.pt"
ann = json.load(open(f"{audit}/manual_annotations_frozen.json"))
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0
def merge(props, thr=0.6):   # identical to V6Pipeline._merge_props
    props = sorted(props, key=lambda p: p["score"], reverse=True); kept = []
    for p in props:
        if all(iou(p["local_box"], q["local_box"]) <= thr for q in kept): kept.append(p)
    return kept
model = YOLO(W0)
from v5.gpu.discovery_v54 import RejectingExpert
expert = RejectingExpert("/workspace/assets", device="cuda:0")
def predict(img, conf, iou_nms=0.7, maxdet=300):
    r = model.predict(img, conf=conf, imgsz=960, iou=iou_nms, max_det=maxdet, verbose=False, device=0)[0]
    return [{"local_box": [float(v) for v in b], "score": float(c)} for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
frames = {}
for f in sorted(glob.glob(f"{audit}/captures/*.png")):
    sc = json.load(open(f[:-4] + ".json")); fi, lv = sc["frame_index"], sc["received_camera"]["resolution_level"]
    img = cv2.imread(f); assert img.shape[:2] == (540, 960)
    t = time.perf_counter(); raw = predict(img, 0.001); ms = (time.perf_counter()-t)*1000
    prod = predict(img, 0.15)
    pm_replay = merge([dict(p, src="ms1") for p in prod])
    pm_stored = sc["candidates"]
    # reproduction check of stored post-merge candidates
    rep = [max([iou(c["local_box"], p["local_box"]) for p in pm_replay] + [0]) for c in pm_stored]
    frames[(fi, lv)] = dict(img=img, sc=sc, raw=raw, pm_replay=pm_replay, ms=ms, repro_min_iou=min(rep) if rep else None,
                            n_stored=len(pm_stored), n_replay=len(pm_replay), n_raw=len(raw))
    print(fi, lv, "raw", len(raw), "prod@.15", len(prod), "stored", len(pm_stored), "replay", len(pm_replay), "repro_min_iou", frames[(fi,lv)]["repro_min_iou"], f"{ms:.0f}ms", flush=True)
res = {"checkpoint": W0, "frames": {}, "targets": []}
for (fi, lv), d in frames.items():
    res["frames"][f"{fi}_{lv}"] = dict(raw=[dict(p, rank=i) for i, p in enumerate(d["raw"])], post_merge_stored=d["sc"]["candidates"],
        post_merge_replay=d["pm_replay"], predictions=d["sc"]["predictions"], region=d["sc"]["received_camera"]["source_region_xyxy"],
        repro_min_iou=d["repro_min_iou"], n_stored=d["n_stored"], n_replay=d["n_replay"], latency_ms=d["ms"])
for t in ann["targets"]:
    d = frames[(t["frame_index"], t["level"])]; gt = t["box_local_xyxy"]; raw = d["raw"]
    ious = [iou(gt, p["local_box"]) for p in raw]
    order = sorted(range(len(raw)), key=lambda i: -ious[i])
    bi = order[0] if raw else None
    r = dict(target_id=t["target_id"], n_raw=len(raw), raw_best_iou=ious[bi] if raw else 0, raw_best_score=raw[bi]["score"] if raw else None,
             raw_best_rank=bi, raw_best_box=raw[bi]["local_box"] if raw else None,
             n_raw_iou30=sum(v >= .3 for v in ious), n_raw_iou50=sum(v >= .5 for v in ious),
             rank_first_iou50=next((i for i, v in enumerate(ious) if v >= .5), None),
             raw_best_iou_conf015=max([v for v, p in zip(ious, raw) if p["score"] >= .15] + [0]))
    for k in (16, 32, 64): r[f"raw_best_iou_top{k}"] = max(ious[:k] + [0])
    sm = [iou(gt, c["local_box"]) for c in d["sc"]["candidates"]]
    r["postmerge_stored_best_iou"] = max(sm + [0]); r["postmerge_stored_n"] = len(sm)
    r["postmerge_stored_best_score"] = d["sc"]["candidates"][int(np.argmax(sm))]["score"] if sm else None
    rm = [iou(gt, c["local_box"]) for c in d["pm_replay"]]; r["postmerge_replay_best_iou"] = max(rm + [0])
    # classification: on best raw proposal (if IoU>=.3), best stored candidate, and the GT box itself (upper bound of the classifier)
    def cls(box):
        if min(box[2]-box[0], box[3]-box[1]) < 4: return None
        try: rs, _, _ = expert.classify(d["img"], [box])
        except Exception as e: return {"error": str(e)}
        q = rs[0]; return {"target_probability": q["target_probability"], "top3": q["top3"], "state": q["state"], "min_side": min(box[2]-box[0], box[3]-box[1])}
    r["cls_gt_box"] = cls([float(v) for v in gt])
    if raw and ious[bi] >= .3: r["cls_best_raw"] = cls(raw[bi]["local_box"])
    if sm and max(sm) >= .3: r["cls_best_postmerge"] = cls(d["sc"]["candidates"][int(np.argmax(sm))]["local_box"])
    # final predictions (normalized global) vs target global box
    g = t["box_source_xyxy"]; pv = []
    for p in d["sc"]["predictions"]:
        b = p["bbox"]; pv.append((iou(g, [b[0]*3840, b[1]*2160, b[2]*3840, b[3]*2160]), p))
    r["final_best_iou_this_frame"] = max([v for v, _ in pv] + [0]); r["final_n_preds_this_frame"] = len(pv)
    r["final_best_pred"] = max(pv, key=lambda z: z[0])[1] if pv else None
    res["targets"].append(r)
json.dump(res, open(f"{audit}/raw_replay_results.json", "w"), indent=1)
print("done")
