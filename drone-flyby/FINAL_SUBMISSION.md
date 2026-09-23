# Final submission — Drone Flyby (Nordic AI Cup 2026)

Authoritative record of exactly what was submitted. Anything not listed here was **not** part of
the submission.

## Official result

| | |
|---|---|
| Score | **0.024644923331075038** |
| Attempt UUID | `c594011b82984f9b9c0cf563ee34e1c9` |
| Type | Final Evaluation (one attempt permitted) |
| Run window | 2026-09-20T13:21:30Z → 13:22:54Z |
| Evaluator errors | 1 — transport timeout at frame 197 |
| Service URL used | `https://4telrb865f5bkf-9053.proxy.runpod.net/predict` |

Best official **validation** score was 0.03182123481701167 (attempt
`a23ca0fbf1124a3c852680c8a39d4d53`). Validation and evaluation use different datasets.
Full record with provenance flags: `architecture_experiments/v7_object_centric/OFFICIAL_RESULTS.json`.

## Submitted source

| | |
|---|---|
| Commit at submission time | `f3c247fb68323f6202a21d19b6f835e2c869a23d` |
| Branch | `claude/hosted-l1l2-audit` (merged into `main` after the competition) |
| Inference entry point | `v5/gpu/endpoint_v6.py` → `app` (FastAPI), route `POST /predict` |
| Pipeline | `v5/gpu/pipeline_v6.py` |
| Serve command | `uvicorn v5.gpu.endpoint_v6:app --host 0.0.0.0 --port 9053 --workers 1` |

`v5/gpu/pipeline_v6.py` integrity:

* sha256 `6f9be14ef9b59875f2b1c6340b321ccb32b31c33489bc9c071a877c865bb83de` — CRLF, as stored on the
  serving host
* sha256 `984303a2153902e7466868fd425dd3d1c9d20bb893a01bf3a5353c3ce5d0e14c` — LF, as stored in Git
* The two differ **only** in line endings; a `diff` after stripping CR is empty.

## Submitted model weights

Not in Git. Verified copies live in the local archive `C:\Nordic-AI-Cup-2026-Final-Archive\checkpoints\`.

| Role | File | sha256 |
|---|---|---|
| **Discovery detector (submitted)** | `v7_e15_snapshot.pt` | `52b9fcef703d3fc5f35993f3088381f545b50d1091c862f52ff0c4df1cd091b8` |
| 16-class visual head | `visual_head.npz` | `b1c04093f1c9b261fc568432817a97302ca8d6441d597b1fdddefaeff3b387d6` |
| Target/background head | `bg_head.npz` | `b1d0f4db02f944615ca7b0e7f3c2dc26dbe6b5e315444974072358882e377910` |
| DINOv2 encoder export | `dinov2_224.onnx` | `ddacda6384a4f5b1a035a976253a1a45a7ae97f6c61cb2099dbc47563099b9ea` |

The detector is a one-class YOLO11s. `v7_e15_snapshot.pt` is the **epoch-15** snapshot of the V7
run; the epoch-30 checkpoint (`v7_final_e30.pt`, `cd370c1f…`) was evaluated and **not** submitted —
the two were within noise of each other on the diagnostic set and the earlier snapshot was kept.

## Runtime configuration

```bash
V6_DETECTOR=<path to v7_e15_snapshot.pt>
V5_ASSETS=<assets dir with dinov2_224.onnx, visual_head.npz, bg_head.npz>
V6_CLASSIFY_MIN_PX=16     # V6 default was 22 — the only gate changed for V7
V6_TARGET_MIN=0.5         # unchanged
V6_DET_CONF=0.15          # unchanged
V6_ENSEMBLE=0             # single detector, no ensemble
V6_DIAGNOSTIC_CAPTURE=0   # diagnostic image capture off
```

Deployment requirements: one CUDA GPU (built and measured on an RTX 4090, ~2 GB VRAM in use),
Python 3.11, `requirements.txt`, a single uvicorn worker (pipeline state is per-sequence and not
safe across independent workers), and a publicly reachable `POST /predict`.

Measured on the submission host: p50 ≈ 53 ms, p95 ≈ 66 ms, max ≈ 70 ms during the Final Evaluation;
a 250-request cadence stress test produced 0 timeouts and no latency growth.

## Differences from the unsuccessful experimental builds

| Build | Change | Official score |
|---|---|---|
| **Submitted V7** | V7 detector, `CLASSIFY_MIN_PX=16` | **0.0246** final / 0.0318 validation |
| V7b camera receive-clamp | planner clamped to moves legal from the received camera | 0.0124 — **rejected** |
| V6 baseline | `ms1` detector, `CLASSIFY_MIN_PX=22` | 0.0178 |

The camera receive-clamp is preserved, unmodified and clearly marked, at
`architecture_experiments/v7_object_centric/NOT_SUBMITTED_pipeline_v6_cameraclamp.py`
(sha256 `ebec47bda72d8b31b739c5573626093cc21e01c86b192d05730bb6f233e77026`). It removed every
camera-legality error yet cost 61% of the score. **Do not deploy it.** The submitted
`v5/gpu/pipeline_v6.py` contains no reference to `V6_RECV_CLAMP`.

## Large assets stored outside Git

| Asset | Location | Why not in Git |
|---|---|---|
| Model checkpoints (9 files) | `…\Final-Archive\checkpoints\` | binary size |
| Generated V7 dataset (3600+420 images) | `…\Final-Archive\datasets\dataset_v7.tgz` | 1.09 GB, regenerable from the seeded generator |
| Hosted diagnostic captures | `…\Final-Archive\diagnostics\ev_diagnostics.tgz` | evaluator imagery; redistribution permission unverified |
| Service and training logs | `…\Final-Archive\logs\logs.tgz` | operational noise |

Manifest with every original→archived path and hash: `…\Final-Archive\MANIFEST.json`.
