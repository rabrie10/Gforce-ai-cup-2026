# Nordic AI Cup 2026 — Team submissions

Our team's entries for the [Nordic AI Cup 2026](https://nordicaicup.com) (17–20 September 2026),
hosted by Ambolt AI. The organiser's original competition brief is preserved in
**[docs/NORDIC_AI_CUP_2026_BRIEF.md](docs/NORDIC_AI_CUP_2026_BRIEF.md)**.

| Use case | Directory | Status |
|---|---|---|
| **Drone Flyby** (main effort) | [`drone-flyby/`](drone-flyby/) | Submitted — Final Evaluation score **0.0246** |
| Survival Simulator | [`survival-simulator/`](survival-simulator/) | Reinforcement-learning agent |
| Medical Appointment | [`medical-appointment/`](medical-appointment/) | — |

---

# Drone Flyby — active-camera aerial object detection

## The problem

A 3840×2160 drone video plays at roughly 3 FPS. For every source frame the service receives
**one 960×540 observation** and must return detections for **16 specialised classes** (aircraft,
helicopters, launchers, vehicles, towers and similar), in **global source-frame coordinates**,
within a **3333 ms** timeout.

The twist is that you choose what you get to see next. Each response may request the camera for the
next frame at one of three zoom levels:

| Level | Source region | Delivered as | Trade-off |
|---|---|---|---|
| **L0** | full 3840×2160 | downscaled to 960×540 | whole scene, 4× detail loss |
| **L1** | 1920×1080 crop | downscaled to 960×540 | half the scene, 2× detail loss |
| **L2** | 960×540 crop | native pixels | full detail, 1/16 of the scene |

Camera moves are constrained: only certain level transitions are legal, and the centre may not move
more than a per-level pixel budget per frame. An illegal request is ignored, and the camera stays
put. So the system has to trade **coverage against resolution** while tracking objects that drift
between frames. Scoring is COCO-style macro mAP@0.50 across the 16 classes.

## Final result

| Metric | Value |
|---|---|
| **Official Final Evaluation score** | **0.024644923331075038** |
| Attempt UUID | `c594011b82984f9b9c0cf563ee34e1c9` |
| Best official validation score | 0.03182123481701167 |
| Previous best (V6 detector) | 0.0178 |

We did **not** place in the top five. Validation and Final Evaluation use different datasets, so
those two numbers are not directly comparable.

## Architecture (V7, as submitted)

```
960×540 observation
   │
   ├─▶ Discovery          one-class YOLO11s  ──▶ candidate boxes
   │                      (V7 detector; ~10 ms on an RTX 4090)
   ├─▶ Merge              greedy cross-source NMS (IoU 0.6)
   ├─▶ Recognition        frozen DINOv2 encoder + 16-class visual head
   ├─▶ Target/background  supervised rejection head, P(target) ≥ 0.5
   ├─▶ Tracking           global-coordinate tracks, per-track velocity +
   │                      scene-drift estimation, TTL 8 frames
   ├─▶ Camera policy      L1 coverage tour + periodic targeted L2 refine,
   │                      legality-guarded against the believed camera state
   └─▶ Emission           per-track class posterior → global normalised boxes
```

Discovery is deliberately **class-agnostic**: one model answers "is this an object worth looking
at", and a separate DINOv2 head decides *what* it is. Measured latency end-to-end: p50 ≈ 53–75 ms,
max ≈ 230 ms, comfortably inside the 3333 ms budget.

## What changed between V6 and V7

V6's detector was fine-tuned on multiscale renderings of 25 reference frames that contain roughly
**one physical instance per class**, always at the same nadir orientation. It scored well locally
and poorly when hosted.

To find out why, we captured 20 genuine hosted L1/L2 observations during a diagnostic run,
annotated the visible targets by hand **before** looking at any model output, and replayed the exact
production checkpoint over them. The detector produced **no usable proposal for 8 of 14** visible
targets — including large, obvious ones at native L2 resolution. The bottleneck was discovery, not
classification or tracking.

V7 keeps the architecture and retrains discovery on an **object-centric, domain-randomised**
distribution: foreground masks are extracted from the training frames (GrabCut, with soft-alpha
matting for the classes GrabCut failed on), then composited onto varied backgrounds with full 360°
rotation, log-uniform scale from 9 to 150 pixels, deliberate edge clipping, occlusion and shadows —
through the evaluator's real L0/L1/L2 view formation. A visibility gate rejects any paste a human
could not see, which prevents training the model to fire on empty ground.

### Measured effect

Three different kinds of number, kept separate on purpose:

| Measurement | Kind | V6 | V7 |
|---|---|---|---|
| Raw proposals at IoU ≥ 0.5 | manually annotated hosted diagnostic, 14 targets | 3/14 | **13/14** |
| End-to-end emitted at IoU ≥ 0.5 | manually annotated hosted diagnostic, 14 targets | 2/14 | **11/14** |
| Helsinki replay mAP@0.50 | local metric, our own harness | 0.240 | 0.247 |
| Hosted score | **official**, organiser-run | 0.0178 | **0.0246** (final) |

The 14-target figures come from **our own manual annotations** on a 20-frame diagnostic sample —
they are a diagnostic gate, not an official score and not an unbiased benchmark. Both helicopters
are localised *and* classified correctly; the tank-like vehicles localise well but receive unstable
class labels, so recognition remains the weak link.

### Negative results worth recording

- **Forcing the camera planner to emit only legal moves lowered the score**, twice: 0.0318 → 0.0124
  for V7, and 0.0178 → 0.0127 earlier for V6. The clamp is *not* in the submitted build.
- **Lowering the detector confidence threshold recovered nothing**, only extra background.
- **One validation scored 0 purely from transport timeouts** — 22 of ~249 requests reached the
  service, each answered in ≤ 236 ms. Infrastructure, not the model.

## Repository layout

| Path | Contents |
|---|---|
| [`drone-flyby/FINAL_SUBMISSION.md`](drone-flyby/FINAL_SUBMISSION.md) | **exact submitted commit, checkpoint, config and entry point** |
| [`drone-flyby/SOLUTION_README.md`](drone-flyby/SOLUTION_README.md) | technical deep-dive and reproduction steps |
| `drone-flyby/v5/gpu/` | submitted inference pipeline and FastAPI endpoint |
| `drone-flyby/architecture_experiments/v7_object_centric/` | V7 mask extraction, dataset generator, training, evaluation harnesses |
| `drone-flyby/architecture_experiments/v6_l1l2_capture/audit/` | hosted failure audit: frozen annotations, failure matrix, overlays |
| `drone-flyby/architecture_experiments/` | 13 earlier experiment tracks, kept for history |
| [`docs/`](docs/) | competition brief, handoff report, submission log |

## Running the final system

Model weights are **not** in Git (see `FINAL_SUBMISSION.md` for their verified local location).

```bash
pip install -r drone-flyby/requirements.txt
export V6_DETECTOR=/path/to/v7_e15_snapshot.pt   # sha256 52b9fcef…a888b8ba
export V5_ASSETS=/path/to/assets                 # DINOv2 + visual/background heads
export V6_CLASSIFY_MIN_PX=16 V6_TARGET_MIN=0.5 V6_DET_CONF=0.15
export V6_ENSEMBLE=0 V6_DIAGNOSTIC_CAPTURE=0
cd drone-flyby && uvicorn v5.gpu.endpoint_v6:app --host 0.0.0.0 --port 9053 --workers 1
```

Score a scene locally: `python local_evaluator.py --url http://127.0.0.1:9053/predict --scene helsinki`

## Tech stack

Python 3.11 · PyTorch · Ultralytics YOLO11 · DINOv2 (ONNX Runtime, CUDA) · FastAPI + Uvicorn ·
OpenCV · NumPy · trained on an RTX 4090.
