"""Evaluator-like continuous stress test.

Mimics the hosted evaluator: one request per frame at a fixed cadence (default 333 ms),
per-request timeout 3333 ms, and on timeout the client ABANDONS that request and moves to the
next frame (exactly what the evaluator does) - so in-flight requests can overlap and pile up.

Records client-side send/receive timestamps, so a delay BEFORE the handler runs (queueing,
body read, proxy) is visible as client_elapsed >> app duration_ms reported by /metrics.
"""
import json, glob, sys, time, argparse, threading
import numpy as np, cv2, requests
sys.path.insert(0, "/workspace/v7-release-candidate")
from utils import encode_image
from dtos import (ALLOWED_RESOLUTION_LEVELS, SOURCE_REGION_SIZES, IMAGE_WIDTH, IMAGE_HEIGHT,
                  MAXIMUM_CENTER_DELTA_PIXELS)

AUD = "/workspace/audit-l1l2"
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:9061/predict")
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--cadence-ms", type=float, default=333.0)
ap.add_argument("--timeout-ms", type=float, default=3333.0)
ap.add_argument("--seq", default="stress-v7")
ap.add_argument("--out", default="/workspace/audit-l1l2/stress_v7.json")
a = ap.parse_args()

caps = []
for f in sorted(glob.glob(f"{AUD}/captures/*.png")):
    sc = json.load(open(f[:-4]+".json")); caps.append((sc, cv2.imread(f)))
caps.sort(key=lambda z: z[0]["frame_index"])
# pre-encode payload images once (encoding cost must not distort the cadence)
enc = [(sc, encode_image(im)) for sc, im in caps]


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


results = [None]*a.n
lock = threading.Lock()
T0 = time.perf_counter()


def fire(i):
    sc, img_b64 = enc[i % len(enc)]
    cam = sc["received_camera"]; lvl = cam["resolution_level"]
    payload = {"sequence_id": a.seq, "frame": i+1, "frame_index": i,
               "request_id": f"{a.seq}:{i}:{lvl}:{cam['center_x']}:{cam['center_y']}",
               "frame_interval_ms": 333, "response_timeout_ms": 3333,
               "original_width": IMAGE_WIDTH, "original_height": IMAGE_HEIGHT,
               "view": {"resolution_level": lvl, "center_x": cam["center_x"], "center_y": cam["center_y"],
                        "view_id": f"v{i}", "image": img_b64, "image_media_type": "image/png",
                        "width": 960, "height": 540, "source_region_xyxy": list(cam["source_region_xyxy"])},
               "camera_constraints": constraints(lvl)}
    sent = time.perf_counter()
    rec = dict(i=i, sent_s=round(sent-T0, 3), status=None, elapsed_ms=None, n_ann=None,
               timeout=False, error=None, requested_view=None, level=lvl,
               given=[lvl, cam["center_x"], cam["center_y"]])
    try:
        r = requests.post(a.url, json=payload, timeout=a.timeout_ms/1000.0)
        rec["elapsed_ms"] = round((time.perf_counter()-sent)*1000, 1)
        rec["status"] = r.status_code
        if r.status_code == 200:
            b = r.json(); rec["n_ann"] = len(b["annotations"]); rec["requested_view"] = b.get("requested_view")
    except requests.exceptions.Timeout:
        rec["timeout"] = True; rec["elapsed_ms"] = round((time.perf_counter()-sent)*1000, 1)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:160]
        rec["elapsed_ms"] = round((time.perf_counter()-sent)*1000, 1)
    with lock:
        results[i] = rec


threads = []
for i in range(a.n):
    due = T0 + i*(a.cadence_ms/1000.0)
    now = time.perf_counter()
    if due > now:
        time.sleep(due-now)
    t = threading.Thread(target=fire, args=(i,), daemon=True); t.start(); threads.append(t)
for t in threads:
    t.join(timeout=10)
wall = time.perf_counter()-T0
res = [r for r in results if r]
ok = [r for r in res if r["status"] == 200]
lat = sorted(r["elapsed_ms"] for r in ok)
to = [r for r in res if r["timeout"]]
err = [r for r in res if r["error"]]


def pct(p):
    return lat[min(len(lat)-1, int(len(lat)*p))] if lat else None


# camera legality of every response
illegal = []
for r in ok:
    rv = r["requested_view"]
    if not rv: continue
    glvl, gx, gy = r["given"]
    d = ((rv["center_x"]-gx)**2 + (rv["center_y"]-gy)**2)**0.5
    w, h = SOURCE_REGION_SIZES[rv["resolution_level"]]
    if (rv["resolution_level"] not in ALLOWED_RESOLUTION_LEVELS[glvl]
            or (rv["resolution_level"] != 0 and d > MAXIMUM_CENTER_DELTA_PIXELS[glvl])
            or not (w//2 <= rv["center_x"] <= IMAGE_WIDTH-w//2)
            or not (h//2 <= rv["center_y"] <= IMAGE_HEIGHT-h//2)):
        illegal.append({"i": r["i"], "given": r["given"], "requested": rv, "delta": round(d, 1),
                        "limit": MAXIMUM_CENTER_DELTA_PIXELS[glvl]})
summary = dict(url=a.url, n=a.n, cadence_ms=a.cadence_ms, wall_s=round(wall, 1),
               ok=len(ok), timeouts=len(to), errors=len(err),
               over_3333=sum(1 for v in lat if v > 3333), over_1000=sum(1 for v in lat if v > 1000),
               over_333=sum(1 for v in lat if v > 333),
               p50=pct(.50), p95=pct(.95), p99=pct(.99), max=lat[-1] if lat else None,
               illegal_camera=len(illegal), illegal_examples=illegal[:5],
               ann_first10=[r["n_ann"] for r in ok[:10]], ann_last10=[r["n_ann"] for r in ok[-10:]],
               lat_first10=[r["elapsed_ms"] for r in ok[:10]], lat_last10=[r["elapsed_ms"] for r in ok[-10:]])
json.dump({"summary": summary, "records": res}, open(a.out, "w"), indent=1)
print(json.dumps(summary, indent=1))
