"""Reproduce the hosted camera stall: the evaluator never applies our commands (because our
responses never arrive), so it keeps reporting the SAME camera while we keep planning moves.
Every requested_view must stay legal FROM THE CAMERA THE EVALUATOR REPORTS."""
import json, glob, sys, argparse
import cv2, requests
sys.path.insert(0, "/workspace/v7-release-candidate")
from utils import encode_image
from dtos import (ALLOWED_RESOLUTION_LEVELS, SOURCE_REGION_SIZES, IMAGE_WIDTH, IMAGE_HEIGHT,
                  MAXIMUM_CENTER_DELTA_PIXELS)
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:9061/predict")
ap.add_argument("--n", type=int, default=25); ap.add_argument("--seq", default="stall-test")
a = ap.parse_args()
f = sorted(glob.glob("/workspace/audit-l1l2/captures/*_152_2_*.png"))[0]
sc = json.load(open(f[:-4]+".json")); im = cv2.imread(f); b64 = encode_image(im)
cam = sc["received_camera"]; LVL, CX, CY = 2, 2001, 1609      # the stuck L2 camera from a23ca0fb


def constraints(level):
    bounds = []
    for t in ALLOWED_RESOLUTION_LEVELS[level]:
        w, h = SOURCE_REGION_SIZES[t]
        bounds.append({"resolution_level": t, "width": 960, "height": 540,
                       "minimum_center_x": w//2, "maximum_center_x": IMAGE_WIDTH-w//2,
                       "minimum_center_y": h//2, "maximum_center_y": IMAGE_HEIGHT-h//2})
    return {"maximum_center_delta": MAXIMUM_CENTER_DELTA_PIXELS[level],
            "allowed_resolution_levels": list(ALLOWED_RESOLUTION_LEVELS[level]),
            "center_bounds": bounds, "full_view_reset_exempt_from_delta": True}


requests.post(a.url.replace("/predict", "/reset"), timeout=20)
illegal, moves = [], 0
for i in range(a.n):
    payload = {"sequence_id": a.seq, "frame": i+1, "frame_index": i,
               "request_id": f"{a.seq}:{i}:{LVL}:{CX}:{CY}", "frame_interval_ms": 333,
               "response_timeout_ms": 3333, "original_width": IMAGE_WIDTH, "original_height": IMAGE_HEIGHT,
               "view": {"resolution_level": LVL, "center_x": CX, "center_y": CY, "view_id": f"v{i}",
                        "image": b64, "image_media_type": "image/png", "width": 960, "height": 540,
                        "source_region_xyxy": list(cam["source_region_xyxy"])},
               "camera_constraints": constraints(LVL)}
    rv = requests.post(a.url, json=payload, timeout=30).json().get("requested_view")
    if rv is None:
        continue
    moves += 1
    d = ((rv["center_x"]-CX)**2 + (rv["center_y"]-CY)**2)**0.5
    w, h = SOURCE_REGION_SIZES[rv["resolution_level"]]
    bad = (rv["resolution_level"] not in ALLOWED_RESOLUTION_LEVELS[LVL]
           or (rv["resolution_level"] != 0 and d > MAXIMUM_CENTER_DELTA_PIXELS[LVL])
           or not (w//2 <= rv["center_x"] <= IMAGE_WIDTH-w//2)
           or not (h//2 <= rv["center_y"] <= IMAGE_HEIGHT-h//2))
    if bad:
        illegal.append({"frame": i, "requested": rv, "delta": round(d, 1),
                        "limit": MAXIMUM_CENTER_DELTA_PIXELS[LVL]})
print(f"frames={a.n} commands_issued={moves} ILLEGAL={len(illegal)}")
for x in illegal[:6]:
    print("   ", x)
print("PASS" if not illegal else "FAIL")
