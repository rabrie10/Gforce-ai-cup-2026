# Frozen Minimal End-to-End Detector Baseline

Measurement date: 2026-09-18  
Branch: `drone-minimal-e2e-baseline`  
Base commit: `0ffe65733aef68f5c0109714e48de88ac2cbbe84`

## Frozen candidate

The measured candidate is YOLO11n trained for 60 epochs at `imgsz=960`, batch 4, seed 20260918. Runtime inference uses class-agnostic NMS, four Torch threads, the always-Level-0 camera policy, no temporal state, and no geometric motion compensation. The confidence threshold and NMS settings were frozen before the final evaluator run. No further detector tuning was performed.

The deployed checkpoint is `models/drone_yolo11n_l0.pt`, 5,494,554 bytes, SHA-256 `4C44E404E03673E8AA15FC85BAF90F2D09B67D90C1FD94C9B6C9A4C1F35204CE`. Its file timestamp and hash remained unchanged during final measurement.

No file under `architecture_experiments/` differs from the reviewed motion baseline commit. The completed translation, similarity, and affine motion-oracle artifacts were not modified.

## Dataset construction

`training/data_prep.py` reconstructs the evaluator's Level-0 view by resizing each 3840x2160 Helsinki source frame to 960x540 using OpenCV `INTER_AREA`, then maps the source GT boxes to normalized YOLO `xywh` labels. All 25 frames and all 259 visible annotations are included. The class map exactly matches the 16 protocol classes. Every normalized box is finite, nondegenerate, and in range.

The deterministic rebuild reproduced `l0_data_summary.json` byte-for-byte with SHA-256 `1ADE5FAA8234380AC86051966C2B20AC5057A8EB44A2141A74F28AF2484E0472`. Three overlay audits at frames 0, 12, and 24 confirmed the resized boxes align with the objects, including clipped boundary objects.

The training and diagnostic validation inputs are the same 25 Helsinki frames. Therefore the Ultralytics training result is a same-frame fit diagnostic, not validation or generalization accuracy. Hosted validation remains the first external test.

## Training setup

The initial weight was the Ultralytics v8.4.0 `yolo11n.pt` asset, SHA-256 `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`. Training used Ultralytics 8.4.155, Torch 2.14.0 CPU, torchvision 0.29.0, Python 3.12.14, deterministic mode, rectangular batches, zero data-loader workers, and the recorded default augmentations. `optimizer=auto` resolved to AdamW with learning rate 0.0005 and momentum 0.9. Wall time was 1,738.504 seconds (28.975 minutes).

The same-frame training diagnostic was mAP50 0.430 and mAP50-95 0.292. These are retained solely as fit diagnostics.

## Final official local evaluation

The final evaluator was run once against the frozen endpoint. Oracle sanity was 1.0 overall and 1.0 for every class.

| Metric | Offline | Realtime |
|---|---:|---:|
| mAP@0.50 | 0.383715501 | 0.125541460 |
| Source frames | 25 | 25 |
| Frames sent | 25 | 8 |
| Responses accepted | 25 | 8 |
| Skipped frames | 0 | 17 |
| Unanswered frames | 0 | 0 |
| Timeouts | 0 | 0 |
| HTTP errors | 0 | 0 |
| Invalid responses | 0 | 0 |
| Camera commands applied / invalid | 0 / 0 | 0 / 0 |
| Total predictions | 235 | 74 |
| Predictions/source frame, mean / median / p95 / max | 9.40 / 10 / 14 / 14 | 2.96 / 0 / 11 / 14 |
| Predictions/sent frame, mean | 9.40 | 9.25 |

### Per-class AP@0.50

| Class | Offline | Realtime |
|---|---:|---:|
| hangar | 0.663366337 | 0.168316832 |
| helicopter | 0.940594059 | 0.316831683 |
| jet_plane | 0.950495050 | 0.277227723 |
| large_launcher | 0.998476771 | 0.326732673 |
| large_tower | 0 | 0 |
| medium_launcher | 0 | 0 |
| medium_plane | 0 | 0 |
| mine_roller | 0 | 0 |
| small_launcher | 0 | 0 |
| small_plane | 0 | 0 |
| small_tower | 0.913248468 | 0.335396040 |
| ta-ta | 0 | 0 |
| tank | 0 | 0 |
| condor | 0.811881188 | 0.277227723 |
| jammer | 0 | 0 |
| spacecraft | 0.861386139 | 0.306930693 |

### Latency in milliseconds

| Timing | Mode | Mean | Median | p95 | Max |
|---|---|---:|---:|---:|---:|
| Model inference | Offline | 178.396 | 173.930 | 187.375 | 307.729 |
| Model predict, including model pre/post-processing | Offline | 188.580 | 182.731 | 202.926 | 320.284 |
| Endpoint processing | Offline | 234.863 | 227.603 | 259.232 | 380.666 |
| HTTP round trip | Offline | 274.960 | 265.000 | 325.000 | 438.000 |
| Model inference | Realtime | 176.205 | 176.295 | 181.986 | 183.111 |
| Model predict, including model pre/post-processing | Realtime | 187.624 | 188.697 | 193.812 | 194.443 |
| Endpoint processing | Realtime | 242.707 | 243.708 | 259.741 | 262.438 |
| HTTP round trip | Realtime | 289.250 | 289.500 | 313.000 | 313.000 |

The realtime evaluator sent frames 0, 3, 6, 9, 12, 15, 18, and 21. Predictions on all eight sent frames exactly equal the predictions from the offline run. Camera state stayed at Level 0 and the endpoint issued no camera commands. The realtime score loss is therefore explained by scheduler accounting: 17 of 25 source frames were skipped and scored without detections. It is not caused by a different camera state or prediction differences. The harness measures its sequential frame loading, Level-0 rendering, PNG encoding, HTTP round trip, and response handling against the realtime clock; endpoint latency alone stayed below 333 ms for all eight realtime responses.

## Direct validation

Five directly callable unit tests pass: exact DTO/class mapping and dataset count; bbox conversion, clipping, deduplication, and degenerate rejection; always-L0 transition legality; exception fallback to a valid empty response; and validity of all emitted annotations. `compileall` passes for the endpoint, detector, and training/evaluation scripts.

The historical local probe suite also completed with exit code 0 from an ordinary shell command in a temporary output location. Its oracle score was 1.0 for all 16 classes and its final integrity flag confirmed the authoritative evaluator inputs were unchanged. No reviewed evaluator-probe artifact was modified.

## Docker readiness

Static build-context inspection passes: the Dockerfile copies the tracked `models/` directory, `.dockerignore` does not exclude it, the local checkpoint is present, and `api.py` loads and warms that local file once at startup. No runtime download path is needed.

The image build itself could not be executed because the installed Docker engine was stopped. Command:

```text
docker build --progress=plain -t drone-flyby-baseline:local .
```

Exact error:

```text
ERROR: error during connect: in the default daemon configuration on Windows, the docker client must be run with elevated privileges to connect: Head "http://%2F%2F.%2Fpipe%2Fdocker_engine/_ping": open //./pipe/docker_engine: The system cannot find the file specified.
```

This is a local Docker service/runtime availability restriction, not a package, model, filesystem, or Dockerfile compatibility failure. Because the engine did not run, image build, container health, representative `/predict`, one-time container load/warmup, and Docker warm-latency remain unverified. The live `:9053` service was not replaced.

## Failure profile

Nine classes have AP=0 in both modes: `large_tower`, `medium_launcher`, `medium_plane`, `mine_roller`, `small_launcher`, `small_plane`, `ta-ta`, `tank`, and `jammer`. There are no positive offline AP values below 0.10; the next weakest class is `hangar` at 0.6634. The strongest offline classes are `large_launcher` (0.9985), `jet_plane` (0.9505), `helicopter` (0.9406), `small_tower` (0.9132), and `spacecraft` (0.8614).

The failures substantially overlap the tiny or rare categories: the dataset contains only two `mine_roller` boxes and five `medium_plane` boxes, while `small_launcher`, `small_plane`, `ta-ta`, and `jammer` are visually small at Level 0. The pattern is not exclusively a tiny-object problem: `large_tower` and `tank` also score zero. Final overlays show reliable localization for the strongest large and distinctive classes, missed tiny targets, boundary misses, and low-confidence false positives. This report diagnoses those failures without attempting to solve them.

## Repository state and changed files

The candidate remains an uncommitted working-tree baseline on `drone-minimal-e2e-baseline`, as requested. Runtime implementation files are `api.py`, `example.py`, and the new `detector.py`; the local checkpoint is under `models/`; reproducibility code is under `training/`; and focused tests are under `tests/`. `Dockerfile`, `.dockerignore`, `.gitignore`, `requirements.txt`, and `README.md` contain packaging and usage support.

Reviewable evidence under `training/artifacts/` includes the data summaries and overlays, training metadata/arguments/results/figures, environment snapshot, final `evaluation_results.json`, prediction overlays, and `validation_results.json`. The detailed machine-readable evaluator predictions and all exact timing rows remain in `evaluation_results.json`.

The exact working-tree file set is:

- Modified: `.gitignore`, `Dockerfile`, `README.md`, `api.py`, `example.py`, `requirements.txt`, `training/data_prep.py`, `training/train.py`.
- Added runtime/package files: `.dockerignore`, `detector.py`, `models/drone_yolo11n_l0.pt`.
- Added test/evaluation/report files: `tests/test_baseline.py`, `training/evaluate_baseline.py`, `training/MINIMAL_E2E_BASELINE.md`.
- Added evidence: `training/artifacts/confusion_matrix_normalized.png`, `evaluation_results.json`, `l0_class_size_summary.csv`, `l0_data_summary.json`, `l0_overlay_frame_000000.jpg`, `l0_overlay_frame_000012.jpg`, `l0_overlay_frame_000024.jpg`, `prediction_overlay_frame_000000.jpg`, `prediction_overlay_frame_000012.jpg`, `prediction_overlay_frame_000024.jpg`, `requirements_environment.txt`, `training_args.yaml`, `training_curves.png`, `training_metadata.json`, `training_results.csv`, and `validation_results.json`.

No commit, merge, deployment, retraining, confidence adjustment, temporal state, GMC, or camera-policy change was performed during final measurement.
