# V5.4 hosted Validation forensics + root cause

First official hosted Validation (attempt `2c59641e01574c2593747fe302dcba43`, commit `dc55f76`):
**official raw score 0.0025**, reported error "Frame 51 did not answer within 3333 ms".

## Annotated folder audit
`drone-flyby/annotated/` = 25 PNGs `frame_000000..024`, ~4.3 MB each. They are **visualization
overlays** of the Helsinki frames 0–24 (colored boxes + class-name text burned into the pixels;
md5 differs from and size ≪ the clean 15.9 MB `src/helsinki/images`). **Not new data, not usable as
training images.** The real machine-readable GT is `src/helsinki/annotations/*.json` (official
classes + 4K-pixel bboxes, one physical instance per class across 25 frames), which V5 already
consumed via `v5/data.py` → `dataset_v52` → detector training. **Verdict: C — not overlooked; it is a
render of data already used.** No new labeled data exists in that folder.

## Endpoint forensics (attempt 2c59641e)
- 142 evaluator `POST /predict` (proxy IPs 100.64.1.x), **all HTTP 200, zero errors/exceptions,
  zero dimension/validation failures**. So the ~0 score is NOT from errors, empty responses, or a
  dimension mismatch. Logs preserved at `logs/endpoint_v54_validation_2c59641e.log`.
- Frame-51 timeout is a real but **secondary** latency event (one frame >3333 ms).

## Root cause (quantified with the OFFICIAL scorer on the Helsinki TRAINING scene; oracle=1.000)
Target sizes: GT boxes 20–166 px at 4K → **5–42 px at L0 (960×540), most only 5–13 px.**
| Config | Official Helsinki mAP@0.50 |
|---|---|
| L0-only (validation config, active_camera OFF) | **0.000** |
| active_camera ON (existing track-driven scheduler) | 0.006 |
| L0 + 3×2 tiling, budget 96 | 0.048 (only `large_launcher`) |

**The dominant failure is that the targets are unresolvable at L0, and the system ran L0-only.**
This is NOT domain gap, overfitting, or the recognition head — even on the exact training instances,
L0-only scores 0.000. Active L1/L2 zoom is mandatory, but the existing track-driven camera scheduler
cannot acquire targets it has never seen (nothing at L0 → nowhere to zoom), and one view per frame vs
16 objects spread across the frame fundamentally caps single-view coverage. Tiling recovers only the
single largest object.

## Next highest-value action
A **systematic high-resolution acquisition strategy** is required (not a config tweak): e.g. a
deterministic L1/L2 scan pattern that sweeps the scene over successive frames to detect the small
objects at native resolution, combined with temporal tracking to retain them in global coordinates
across frames when the view moves on. This is a substantial change with an uncertain ceiling given the
one-view-per-frame scoring, and should be prototyped and measured against this offline official scorer
(now available on the pod) BEFORE spending another hosted Validation. Latency (proxy ~0.7 s; realtime
frame-skip) is a secondary concern to address once detection is nonzero.
