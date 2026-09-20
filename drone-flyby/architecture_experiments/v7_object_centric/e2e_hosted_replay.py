"""End-to-end replay of the 20 preserved genuine hosted observations through a running
V6/V7 endpoint (isolated port). Evaluation-only use of the hosted captures.

Two modes:
  --fresh  each observation as its own sequence (clean per-observation attribution:
           discovery -> classify -> track create -> emit, with no bogus association
           between non-contiguous frames)
  --seq    all 20 in frame order as one sequence (track proliferation / camera / memory check)

Reports per frozen target: final emitted class, confidence, global bbox and IoU against the
frozen manual box; plus per-frame false positives, camera-command legality and latency.
"""
import json, glob, sys, time, argparse
import numpy as np, cv2, requests
sys.path.insert(0, "/workspace/v7-release-candidate")
from utils import encode_image
from dtos import (ALLOWED_RESOLUTION_LEVELS, SOURCE_REGION_SIZES, IMAGE_WIDTH, IMAGE_HEIGHT,
                  MAXIMUM_CENTER_DELTA_PIXELS)

AUD = "/workspace/audit-l1l2"
PAD = 1.25


def iou(a, b):
    ix = max(0, min(a[2], b[2])-max(a[0], b[0])); iy = max(0, min(a[3], b[3])-max(a[1], b[1])); i = ix*iy
    u = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u > 0 else 0.


def padb(b, s):
    cx, cy, w, h = (b[0]+b[2])/2, (b[1]+b[3])/2, (b[2]-b[0])*s, (b[3]-b[1])*s
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


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


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:9061/predict")
ap.add_argument("--mode", choices=["fresh", "seq"], default="fresh")
ap.add_argument("--out", default="/workspace/audit-l1l2/e2e_v7.json")
a = ap.parse_args()

ann = json.load(open(f"{AUD}/manual_annotations_frozen.json"))
caps = []
for f in sorted(glob.glob(f"{AUD}/captures/*.png")):
    sc = json.load(open(f[:-4]+".json")); caps.append((sc, cv2.imread(f)))
caps.sort(key=lambda z: z[0]["frame_index"])

requests.post(a.url.replace("/predict", "/reset"), timeout=20)
out = {"mode": a.mode, "frames": {}, "latency_ms": [], "illegal_camera": [], "errors": []}
SEQ = "v7-e2e-offline"
for sc, im in caps:
    fi = sc["frame_index"]; cam = sc["received_camera"]; lvl = cam["resolution_level"]
    payload = {"sequence_id": (f"{SEQ}-{fi}" if a.mode == "fresh" else SEQ),
               "frame": sc.get("frame", fi), "frame_index": fi,
               "request_id": f"{SEQ}:{fi}", "frame_interval_ms": 333, "response_timeout_ms": 3333,
               "original_width": IMAGE_WIDTH, "original_height": IMAGE_HEIGHT,
               "view": {"resolution_level": lvl, "center_x": cam["center_x"], "center_y": cam["center_y"],
                        "view_id": f"v{fi}", "image": encode_image(im), "image_media_type": "image/png",
                        "width": 960, "height": 540, "source_region_xyxy": list(cam["source_region_xyxy"])},
               "camera_constraints": constraints(lvl)}
    t = time.perf_counter()
    r = requests.post(a.url, json=payload, timeout=30)
    ms = (time.perf_counter()-t)*1000
    if r.status_code != 200:
        out["errors"].append({"frame": fi, "status": r.status_code, "body": r.text[:300]}); continue
    body = r.json()
    out["latency_ms"].append(ms)
    rv = body.get("requested_view")
    if rv is not None:                       # legality vs the camera we were actually given
        ok_lvl = rv["resolution_level"] in ALLOWED_RESOLUTION_LEVELS[lvl]
        d = ((rv["center_x"]-cam["center_x"])**2 + (rv["center_y"]-cam["center_y"])**2)**0.5
        ok_d = rv["resolution_level"] == 0 or d <= MAXIMUM_CENTER_DELTA_PIXELS[lvl]
        w, h = SOURCE_REGION_SIZES[rv["resolution_level"]]
        ok_b = (w//2 <= rv["center_x"] <= IMAGE_WIDTH-w//2) and (h//2 <= rv["center_y"] <= IMAGE_HEIGHT-h//2)
        if not (ok_lvl and ok_d and ok_b):
            out["illegal_camera"].append({"frame": fi, "given": [lvl, cam["center_x"], cam["center_y"]],
                                          "requested": rv, "level_ok": ok_lvl, "delta_ok": ok_d, "bounds_ok": ok_b})
    out["frames"][str(fi)] = {"level": lvl, "latency_ms": round(ms, 1), "requested_view": rv,
                              "annotations": body["annotations"], "n_ann": len(body["annotations"])}

rows = []
for t in ann["targets"]:
    fi = t["frame_index"]; fr = out["frames"].get(str(fi))
    g = t["box_source_xyxy"]
    best = (0., None)
    if fr:
        for p in fr["annotations"]:
            b = p["bbox"]; gb = [b[0]*IMAGE_WIDTH, b[1]*IMAGE_HEIGHT, b[2]*IMAGE_WIDTH, b[3]*IMAGE_HEIGHT]
            v = max(iou(g, gb), iou(padb(g, PAD), gb))
            if v > best[0]: best = (v, dict(p, global_box=[round(x, 1) for x in gb]))
    rows.append(dict(target=t["target_id"], frame=fi, level=t["level"], family=t["family"],
                     difficulty=t["difficulty"], manual_conf=t["existence_confidence"],
                     final_iou=round(best[0], 3), emitted=best[1]))
out["targets"] = rows

# false positives: emitted predictions that overlap no frozen target in that frame
tg = {}
for t in ann["targets"]: tg.setdefault(t["frame_index"], []).append(t["box_source_xyxy"])
fp = 0; tot = 0
for fi, fr in out["frames"].items():
    for p in fr["annotations"]:
        b = p["bbox"]; gb = [b[0]*IMAGE_WIDTH, b[1]*IMAGE_HEIGHT, b[2]*IMAGE_WIDTH, b[3]*IMAGE_HEIGHT]
        tot += 1
        if max([iou(gb, x) for x in tg.get(int(fi), [])] + [0]) < 0.2: fp += 1
out["emitted_total"] = tot; out["emitted_fp"] = fp
json.dump(out, open(a.out, "w"), indent=1)

L = sorted(out["latency_ms"])
print(f'MODE {a.mode}  frames {len(out["frames"])}  errors {len(out["errors"])}  illegal_camera {len(out["illegal_camera"])}')
if L: print(f'LATENCY ms  p50 {L[len(L)//2]:.0f}  p95 {L[int(len(L)*0.95)-1]:.0f}  max {L[-1]:.0f}')
print(f'EMITTED total {tot}  no-overlap-with-frozen-target {fp}  ({tot/max(1,len(out["frames"])):.1f}/frame)')
print(f'{"target":20s} {"diff":8s} {"conf":5s} | final_iou | emitted class / confidence')
for r in rows:
    e = r["emitted"]
    print(f'{r["target"]:20s} {r["difficulty"]:8s} {r["manual_conf"][:4]:5s} | {r["final_iou"]:9.2f} | '
          f'{(e["object_id"]+" "+format(e["confidence"], ".3f")) if e and r["final_iou"] > 0 else "-- none --"}')
n50 = sum(r["final_iou"] >= .5 for r in rows); n30 = sum(r["final_iou"] >= .3 for r in rows)
print(f'FINAL EMISSION IoU>=0.5 (incl. pad-sensitivity): {n50}/14   IoU>=0.3: {n30}/14')
