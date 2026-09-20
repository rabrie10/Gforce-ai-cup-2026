# Drone Flyby — V7 solution (Nordic AI Cup 2026)

**Final Evaluation score: 0.024644923331075038** (attempt `c594011b82984f9b9c0cf563ee34e1c9`, 2026-09-20).
Best validation score: 0.03182123481701167 (attempt `a23ca0fbf1124a3c852680c8a39d4d53`).
Previous best with the V6 detector: 0.0178.

This document describes the submitted system. The organiser's repository README is unchanged at
the repo root; this file covers our solution only.

## What is submitted

| item | value |
|---|---|
| Git commit | `f3c247fb68323f6202a21d19b6f835e2c869a23d` (branch `claude/hosted-l1l2-audit`) |
| Discovery detector | `v7_e15_snapshot.pt`, sha256 `52b9fcef703d3fc5f35993f3088381f545b50d1091c862f52ff0c4df1cd091b8` |
| Serving code | `v5/gpu/endpoint_v6.py` + `v5/gpu/pipeline_v6.py` (sha256 `6f9be14e…`) |
| Runtime config | `V6_CLASSIFY_MIN_PX=16`, `V6_TARGET_MIN=0.5`, `V6_DET_CONF=0.15`, `V6_ENSEMBLE=0`, `V6_DIAGNOSTIC_CAPTURE=0` |

> **Not submitted:** the camera receive-clamp experiment (`V6_RECV_CLAMP`). It is absent from the
> submitted `pipeline_v6.py`. A copy of the clamp variant is archived locally as
> `experiments/NOT_SUBMITTED_pipeline_v6_cameraclamp.py`. See "Negative results" below.

## Architecture

Unchanged from V6 except the discovery detector and one gate:

1. **Discovery** — one-class YOLO11s (`ms1` in V6, **V7 in the submission**) on the 960x540 view.
2. **Merge** — greedy cross-source NMS (IoU 0.6).
3. **Recognition** — frozen DINOv2 (`dinov2_224.onnx`) + 16-class visual head (`visual_head.npz`).
4. **Target/background rejection** — supervised head (`bg_head.npz`), P(target) >= `V6_TARGET_MIN`.
5. **Tracking** — global-coordinate tracks with velocity + scene-drift estimation, TTL 8.
6. **Camera control** — L1 coverage tour + periodic targeted L2 refine, legality-guarded.
7. **Emission** — per-track class posterior, global normalised boxes.

Only proposals whose box short side is >= `V6_CLASSIFY_MIN_PX` (16) are classified.

## V7 discovery detector — what changed and why

The V6 detector was fine-tuned on multiscale renderings of 25 Helsinki frames, which contain
roughly **one physical instance per class** at a fixed nadir orientation. A forensic audit of 20
genuine hosted L1/L2 observations (`architecture_experiments/v6_l1l2_capture/audit/`) showed the
detector produced **no usable proposal for 8 of 14** visible targets, including large, obvious ones
at native L2 resolution.

V7 keeps the same architecture and retrains discovery on an **object-centric, domain-randomised**
distribution (`architecture_experiments/v7_object_centric/gen_v7_dataset.py`):

* Foreground masks extracted from the Helsinki frames (GrabCut, plus soft-alpha matting for the
  five classes where GrabCut failed) — `extract_masks*.py`.
* Full 360-degree in-plane rotation, log-uniform scale from 9 to 150 observation pixels,
  random position, deliberate edge clipping, partial occlusion, drop shadows.
* Genuine evaluator-style L0/L1/L2 view formation, with the official Helsinki boxes retained,
  so the dataset is a superset of the V6 distribution.
* A **visibility gate** rejects any paste that is not visibly different from the background,
  preventing invisible "ghost" positives.

3600 train / 420 val images, ~25k boxes, seed 20260920. Training: `train_v7.py`
(YOLO11s from COCO, imgsz 960, single-class, AdamW, 21-minute wall-clock cap; the submitted
checkpoint is the epoch-15 snapshot).

### Measured effect (frozen hosted annotations, evaluation-only)

| metric | V6 `ms1` | V7 |
|---|---|---|
| raw proposal IoU >= 0.5 | 3/14 | **13/14** |
| distinct physical objects recovered | 3 | **8** |
| end-to-end emitted IoU >= 0.5 | 2/14 | **11/14** |
| detector latency | 10.0 ms | 10.3 ms |

Both helicopters are localised **and** classified correctly. The tank-like vehicles localise
correctly but receive unstable classes (`small_tower`, `ta-ta`, `jammer`) — recognition, not
discovery, is the remaining bottleneck.

## Negative results worth keeping

1. **Strict camera legality reduces the score.** Clamping the planner so every requested move is
   legal from the received camera removed all camera errors but dropped the score
   **0.0318 -> 0.0124**. The same pattern appeared in V6 (0.0178 -> 0.0127 after a legality
   correction). Do not "fix" camera legality without measuring coverage.
2. **Lowering `V6_DET_CONF` buys nothing.** Gate ablation recovered no additional targets at 0.10
   or 0.05, only extra background.
3. **Transport, not compute, caused one zero-scoring attempt.** In attempt `7306c67f` only 22 of
   ~249 evaluator requests reached the application; each was answered in <= 236 ms.

## Reproducing

```bash
# 1. masks from the Helsinki training assets
python architecture_experiments/v7_object_centric/extract_masks.py
python architecture_experiments/v7_object_centric/extract_masks2.py
python architecture_experiments/v7_object_centric/extract_masks3.py
python architecture_experiments/v7_object_centric/extract_masks4.py   # final soft mattes

# 2. dataset (6 shards x 600 train / 70 val, seeds 20260920..20260925)
for i in 0 1 2 3 4 5; do
  python architecture_experiments/v7_object_centric/gen_v7_dataset.py \
      --tag s$i --seed $((20260920+i)) --n-train 600 --n-val 70
done

# 3. train
python architecture_experiments/v7_object_centric/train_v7.py

# 4. evaluate against the frozen hosted annotations
python architecture_experiments/v7_object_centric/compare_v7.py <candidate.pt>
python architecture_experiments/v7_object_centric/gate_ablation.py <candidate.pt>
```

Paths inside these scripts are absolute to the Runpod host (`/workspace/...`) and must be adapted.
Bit-identical regeneration has **not** been verified; the generated dataset is archived instead.

## Serving

```bash
export V6_DETECTOR=/path/to/v7_e15_snapshot.pt
export V6_CLASSIFY_MIN_PX=16 V6_TARGET_MIN=0.5 V6_DET_CONF=0.15
export V6_ENSEMBLE=0 V6_DIAGNOSTIC_CAPTURE=0 V5_ASSETS=/path/to/assets
uvicorn v5.gpu.endpoint_v6:app --host 0.0.0.0 --port 9053 --workers 1
```

Measured: p50 ~53-75 ms, p95 ~105 ms, max ~230 ms on an RTX 4090, against a 3333 ms evaluator
timeout. 250-request cadence stress: 0 timeouts, no latency growth, stable memory.

## Large assets not in Git

Model weights, the generated dataset and the hosted diagnostic captures live in the local archive
`Nordic-AI-Cup-2026-Final-Archive` (see `FINAL_PROJECT_HANDOFF.md`), not in this repository.
