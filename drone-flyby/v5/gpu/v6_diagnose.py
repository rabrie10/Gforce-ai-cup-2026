"""Offline V6 diagnosis (GT used for analysis only, never fed to pipeline):
P1 camera legality (exact refused transitions) + P3 per-class first-failure stage A/B/C/D."""
import os, json, numpy as np
from dtos import DroneFlybyPredictRequestDto
from local_evaluator import Camera, render_view, build_request, CameraRejection
from utils import load_frame, load_annotations, frame_numbers, global_bbox_to_source
from v2.geometry import iou
from v5.gpu.pipeline_v6 import V6Pipeline, Config

DOWN = {0: 4, 1: 2, 2: 1}
pipe = V6Pipeline(Config())
props_by_frame = {}
_orig = pipe._detect
def _rec(view, region):
    out = _orig(view, region)
    props_by_frame[_cur[0]] = [p["source_box"] for p in out]
    return out
pipe._detect = _rec
_cur = [None]

frames = frame_numbers("helsinki")
camera = Camera(); feedback = None
refused = []
# per class per frame stage flags
classes = set()
seen = {}  # cls -> dict of stage bools aggregated
def ensure(cls):
    seen.setdefault(cls, {"visible": 0, "proposal": 0, "track": 0, "class_ok": 0, "emitted": 0, "appearances": 0})

seq = "local"  # local_evaluator SEQUENCE_ID
for idx, frame in enumerate(frames):
    _cur[0] = frame
    img = load_frame(frame, "helsinki")
    anns = load_annotations(frame, "helsinki")
    payload = build_request(frame, idx, camera, render_view(img, camera), feedback)
    resp = pipe.predict(DroneFlybyPredictRequestDto.model_validate(payload))
    region = camera.source_region
    tracks = pipe.states[seq].tracks if seq in pipe.states else []
    emitted = [(a.object_id, global_bbox_to_source(a.bbox, 3840, 2160)) for a in resp.annotations]
    props = props_by_frame.get(frame, [])
    for a in anns:
        cls = a["object_id"]; gb = a["bbox"]; ensure(cls); classes.add(cls)
        seen[cls]["appearances"] += 1
        gw, gh = gb[2]-gb[0], gb[3]-gb[1]
        inview = min(region[2], gb[2]) > max(region[0], gb[0]) and min(region[3], gb[3]) > max(region[1], gb[1])
        vis = inview and (min(gw, gh) / DOWN[camera.resolution_level]) >= 8
        if vis:
            seen[cls]["visible"] += 1
        if any(iou(gb, p) >= 0.5 for p in props):
            seen[cls]["proposal"] += 1
        tk = [t for t in tracks if iou(gb, t.box()) >= 0.3]
        if tk:
            seen[cls]["track"] += 1
            if any(int(t.ev.argmax()) < 16 and __import__("dtos").OBJECT_CLASSES[int(t.ev.argmax())] == cls and t.ev.sum() > 0 for t in tk):
                seen[cls]["class_ok"] += 1
        if any(lbl == cls and iou(gb, sb) >= 0.5 for lbl, sb in emitted):
            seen[cls]["emitted"] += 1
    # camera step with real legality check
    if resp.requested_view is not None:
        rv = resp.requested_view
        prev = (camera.resolution_level, camera.center_x, camera.center_y)
        try:
            camera.apply(rv.resolution_level, rv.center_x, rv.center_y); feedback = None
        except CameraRejection as e:
            refused.append({"frame": frame, "prev": prev, "requested": [rv.resolution_level, rv.center_x, rv.center_y], "reason": str(e)})
            feedback = {"frame": frame, "requested_view": {"resolution_level": rv.resolution_level, "center_x": rv.center_x, "center_y": rv.center_y}, "reason": str(e)}

print("=== REFUSED CAMERA MOVES ===", len(refused))
for r in refused:
    print(r)
print("\n=== PER-CLASS FIRST FAILURE (A=coverage,B=no proposal,C=reject/misclass,D=track/coord/propagate) ===")
OC = __import__("dtos").OBJECT_CLASSES
for cls in OC:
    s = seen.get(cls)
    if not s:
        continue
    if s["visible"] == 0:
        stage = "A_no_resolvable_view"
    elif s["proposal"] == 0:
        stage = "B_no_proposal"
    elif s["class_ok"] == 0:
        stage = "C_reject_or_misclass"
    elif s["emitted"] == 0:
        stage = "D_track/coord/propagation"
    else:
        stage = "OK"
    print(f"{cls:16s} {stage:26s} vis={s['visible']}/{s['appearances']} prop={s['proposal']} trk={s['track']} clsok={s['class_ok']} emit={s['emitted']}")
json.dump({"refused": refused, "per_class": seen}, open("/workspace/results/v6_diagnose.json", "w"), indent=2, default=int)
print("final_global_drift", pipe.states.get(seq).gvx if seq in pipe.states else None, pipe.states.get(seq).gvy if seq in pipe.states else None); print("WROTE /workspace/results/v6_diagnose.json")
