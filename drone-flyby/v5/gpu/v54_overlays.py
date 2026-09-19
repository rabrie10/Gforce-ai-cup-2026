import os, glob, json, cv2, numpy as np
from dtos import DroneFlybyPredictRequestDto
from local_evaluator import Camera, build_request
from utils import encode_image
from v5.pipeline import Pipeline, Config
from v5.gpu.discovery_v54 import MergedDiscovery, RejectingExpert

OUT = "/workspace/results/v54_overlays"; os.makedirs(OUT, exist_ok=True)
HOSTED = "/workspace/hosted-v3"
manual = json.load(open("/workspace/results/manual_targets.json"))["targets"]
mby = {}
for t in manual:
    mby.setdefault(t["capture"], []).append(t)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.


def draw(image, boxes, caption, color=(0, 220, 255)):
    c = image.copy()
    for box, label in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(c, (x1, y1), (x2, y2), color, 1)
        if label:
            cv2.putText(c, label, (max(0, min(750, x1)), max(12, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(c, label, (max(0, min(750, x1)), max(12, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1, cv2.LINE_AA)
    hdr = np.full((42, 960, 3), 24, np.uint8)
    cv2.putText(hdr, caption, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([hdr, c], axis=0)


cfg = Config(temporal=False, active_camera=False)
disc = MergedDiscovery(cfg.assets, budget=cfg.candidate_budget, device=cfg.device)
expert = RejectingExpert(cfg.assets, device=cfg.device, head_name=cfg.head_name)
pipe = Pipeline(cfg, discovery=disc, expert=expert)

indices = [60, 125, 185, 245]
summary = []
for idx in indices:
    matches = sorted(glob.glob(f"{HOSTED}/images/{idx:05d}*.png"))
    if not matches:
        continue
    path = matches[0]
    meta_p = f"{HOSTED}/frames/" + os.path.basename(path)[:-4] + ".json"
    meta = json.loads(open(meta_p).read()) if os.path.exists(meta_p) else {}
    src = meta.get("frame", idx)
    img = cv2.imread(path)
    payload = build_request(src, idx, Camera(), encode_image(img), None)
    payload["sequence_id"] = f"v54-{idx}"
    resp = pipe.predict(DroneFlybyPredictRequestDto.model_validate(payload))
    diag = pipe.last_diagnostics
    raw = diag["candidates"]; sel = [c for c in raw if c["selected"]]
    final = [([b*s for b, s in zip(a.bbox, (960, 540, 960, 540))], f"{a.object_id} {a.confidence:.2f}") for a in resp.annotations]
    rawb = [(c["local_box"], "") for c in raw[:300]]
    selb = [(c["local_box"], f'{c["top3"][0]["class"]}{c["target_probability"]:.1f}' if c.get("top3") else "") for c in sel]
    man = mby.get(idx, [])
    manb = [(t["box"], t["id"].replace("t"+str(idx), "")) for t in man]
    panels = [draw(img, [], f"A idx{idx} src{src}"),
              draw(img, rawb, f"C v5.4 raw proposals {len(raw)}"),
              draw(img, selb, f"D selected {len(sel)}"),
              draw(img, final, f"E emitted {len(final)}", (70, 240, 70)),
              draw(img, manb, "F manual diagnostic (approx)", (0, 255, 0))]
    top = np.concatenate(panels[:3], axis=1) if len(panels) >= 3 else panels[0]
    # 2 rows: A C D | E F (pad)
    row1 = np.concatenate([panels[0], panels[1], panels[2]], axis=1)
    blank = np.zeros_like(panels[0])
    row2 = np.concatenate([panels[3], panels[4], blank], axis=1)
    cv2.imwrite(f"{OUT}/v54_{idx:05d}.png", np.concatenate([row1, row2], axis=0))
    # per-target recovery (only frames with manual boxes)
    rec = []
    for t in man:
        mb = t["box"]
        braw = max((iou(mb, c["local_box"]) for c in raw), default=0.0)
        bsel = max((iou(mb, c["local_box"]) for c in sel), default=0.0)
        bem = max(((iou(mb, fb), lbl) for fb, lbl in final), default=(0.0, ""))
        rec.append({"id": t["id"], "best_raw_iou": round(braw, 3), "best_sel_iou": round(bsel, 3),
                    "best_emitted_iou": round(bem[0], 3), "emitted_label": bem[1] if bem[0] > 0.1 else None})
    summary.append({"idx": idx, "src": src, "raw": len(raw), "selected": len(sel), "emitted": len(final),
                    "timing_ms": round(diag["timing"]["total_ms"], 1), "targets": rec})
    print(f"idx{idx} src{src}: raw={len(raw)} sel={len(sel)} emit={len(final)} t={diag['timing']['total_ms']:.0f}ms")
    for r in rec:
        print(f"    {r['id']}: raw={r['best_raw_iou']} sel={r['best_sel_iou']} emit={r['best_emitted_iou']} ({r['emitted_label']})")
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
print("WROTE", OUT)
