"""Run V6 in-process through the Helsinki camera loop (causal, honoring requested_view);
draw full-source overlays: GT (white), camera region (cyan), emitted predictions (green),
detector proposals (yellow). Diagnostic only."""
import os, cv2, numpy as np
from dtos import DroneFlybyPredictRequestDto
from local_evaluator import Camera, render_view, build_request, CameraRejection
from utils import load_frame, load_annotations, frame_numbers, global_bbox_to_source
from v5.gpu.pipeline_v6 import V6Pipeline, Config

os.environ.setdefault("V6_TTL", "15")
OUT = "/workspace/results/v6_overlays"; os.makedirs(OUT, exist_ok=True)
pipe = V6Pipeline(Config())
SAVE = {4, 10, 16, 22}
frames = frame_numbers("helsinki")
camera = Camera(); feedback = None
S = 0.35  # display scale


def draw():
    canvas = cv2.resize(img, (int(3840*S), int(2160*S)))
    for a in anns:  # GT white
        b = [int(v*S) for v in a["bbox"]]
        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 1)
    r = camera.source_region  # cyan camera region
    cv2.rectangle(canvas, (int(r[0]*S), int(r[1]*S)), (int(r[2]*S), int(r[3]*S)), (255, 255, 0), 2)
    for p in pipe.last_diagnostics.get("props_src", []):  # proposals yellow
        b = [int(v*S) for v in p]
        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), (0, 220, 255), 1)
    for a in resp.annotations:  # emitted green
        gb = global_bbox_to_source(a.bbox, 3840, 2160)
        b = [int(v*S) for v in gb]
        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), (0, 240, 70), 2)
        cv2.putText(canvas, f"{a.object_id} {a.confidence:.2f}", (b[0], max(10, b[1]-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 240, 70), 1, cv2.LINE_AA)
    hdr = np.full((30, canvas.shape[1], 3), 24, np.uint8)
    d = pipe.last_diagnostics
    cv2.putText(hdr, f"frame {frame} idx {idx} L{camera.resolution_level}({camera.center_x},{camera.center_y}) "
                f"props {d.get('n_props')} tracks {d.get('n_tracks')} emit {d.get('n_emitted')} drift {d.get('global_drift')}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([hdr, canvas], axis=0)


# monkeypatch pipeline to record source proposals for the overlay
_orig = pipe._detect
def _rec(view, region):
    out = _orig(view, region)
    pipe.last_diagnostics["props_src"] = [p["source_box"] for p in out]
    return out
pipe._detect = _rec

for idx, frame in enumerate(frames):
    img = load_frame(frame, "helsinki")
    anns = load_annotations(frame, "helsinki")
    payload = build_request(frame, idx, camera, render_view(img, camera), feedback)
    resp = pipe.predict(DroneFlybyPredictRequestDto.model_validate(payload))
    if frame in SAVE:
        cv2.imwrite(f"{OUT}/v6_frame_{frame:02d}.png", draw())
    if resp.requested_view is not None:
        try:
            camera.apply(resp.requested_view.resolution_level, resp.requested_view.center_x, resp.requested_view.center_y)
            feedback = None
        except CameraRejection as e:
            feedback = {"frame": frame, "requested_view": {"resolution_level": resp.requested_view.resolution_level,
                        "center_x": resp.requested_view.center_x, "center_y": resp.requested_view.center_y}, "reason": str(e)}
print("WROTE", OUT, sorted(SAVE))
