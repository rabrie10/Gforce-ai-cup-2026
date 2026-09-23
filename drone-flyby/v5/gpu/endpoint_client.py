"""Contract + latency client for the isolated V5 endpoint. Builds official DTOs,
validates the competition response contract, measures latency."""
import os, glob, json, time, math
import numpy as np, cv2, requests
from local_evaluator import Camera, build_request
from utils import encode_image
from dtos import OBJECT_CLASSES

URL = os.environ.get("V5_URL", "http://127.0.0.1:9053")
HOSTED = "/workspace/hosted-v3"
CLASSES = set(OBJECT_CLASSES)


def frames():
    out = []
    for p in sorted(glob.glob(HOSTED + "/images/*.png")):
        idx = int(os.path.basename(p)[:5])
        meta_p = HOSTED + "/frames/" + os.path.basename(p)[:-4] + ".json"
        meta = json.loads(open(meta_p).read()) if os.path.exists(meta_p) else {}
        out.append((idx, p, meta.get("frame", idx)))
    return out


def validate(idx, req_payload, resp):
    v = []
    if resp.get("request_id") != req_payload["request_id"]:
        v.append("request_id mismatch")
    if resp.get("frame") != req_payload["frame"]:
        v.append("frame mismatch")
    ann = resp.get("annotations", [])
    if len(ann) > 500:
        v.append(f"annotations>{500}: {len(ann)}")
    for a in ann:
        if a["object_id"] not in CLASSES:
            v.append(f"bad class {a['object_id']}"); break
        b = a["bbox"]
        if len(b) != 4 or not all(math.isfinite(x) for x in b):
            v.append("nonfinite bbox"); break
        if not all(0.0 <= x <= 1.0 for x in b):
            v.append(f"bbox out of [0,1]: {b}"); break
        if not (b[0] < b[2] and b[1] < b[3]):
            v.append(f"degenerate bbox: {b}"); break
        if not (0.0 <= a["confidence"] <= 1.0):
            v.append("confidence out of range"); break
    rv = resp.get("requested_view")
    return v, rv


def main():
    requests.post(URL + "/reset")
    fr = frames()
    lat = []
    viol = 0
    seq = "endpoint-contract-seq"
    emitted_total = 0
    for i, (idx, path, srcframe) in enumerate(fr):
        img = cv2.imread(path)
        payload = build_request(srcframe, idx, Camera(), encode_image(img), None)
        payload["sequence_id"] = seq  # feed as one causal sequence
        t = time.perf_counter()
        r = requests.post(URL + "/predict", json=payload, timeout=30)
        dt = (time.perf_counter() - t) * 1000
        lat.append(dt)
        resp = r.json()
        v, rv = validate(idx, payload, resp)
        emitted_total += len(resp.get("annotations", []))
        if v:
            viol += 1
            if viol <= 5:
                print(f"  CONTRACT VIOLATION frame {idx}: {v}")
    lat = np.array(lat)
    print(f"frames={len(fr)} contract_violations={viol} total_emitted={emitted_total}")
    print(f"HTTP latency ms: p50={np.percentile(lat,50):.1f} p95={np.percentile(lat,95):.1f} max={lat.max():.1f} mean={lat.mean():.1f}")
    m = requests.get(URL + "/metrics").json()
    print("server /metrics endpoint_ms:", m.get("endpoint_ms"), "budget:", m.get("configuration", {}).get("candidate_budget"), "device:", m.get("configuration", {}).get("device"))
    # frame-cadence skip estimate at 333 ms
    skipped = int((lat > 333).sum())
    print(f"frames over 333ms cadence: {skipped}/{len(fr)} ({100*skipped/len(fr):.0f}%)  over 3333ms timeout: {int((lat>3333).sum())}")
    json.dump({"n": len(fr), "violations": viol, "p50": float(np.percentile(lat,50)), "p95": float(np.percentile(lat,95)),
               "max": float(lat.max()), "over_333ms": skipped, "note": "GPU shared with DA training if running -> latency inflated"},
              open("/workspace/results/endpoint_contract.json", "w"), indent=2)


if __name__ == "__main__":
    main()
