# V5 GPU deployment (Runpod RTX 4090)

Extends the existing V5 (`a74d419`) to run on an NVIDIA RTX 4090. No V5 model code
was rewritten; `discovery.py`/`visual_expert.py` already support `device=cuda:0`
via ONNX Runtime `CUDAExecutionProvider` with a fail-fast guard (kept).

## Pod / environment
- Runpod pod `ready_azure_guanaco`, RTX 4090 24 GB, driver 580, torch 2.8.0+cu128 (CUDA verified).
- `ssh -i ~/.ssh/runpod_nordic_2026 root@213.181.111.2 -p 51567` (IP/port can change on relocation).
- Python env: `/workspace/venv` (`python -m venv --system-site-packages`, base env untouched)
  with `onnxruntime-gpu==1.20.2`, `ultralytics==8.4.155`, `onnx==1.20.1`, `scikit-learn==1.7.2`,
  `opencv-python-headless`, `fastapi`, `uvicorn`, `pydantic>=2`.
- **ONNX Runtime CUDA needs the torch-bundled NVIDIA libs on `LD_LIBRARY_PATH`.** `/workspace/env.sh`
  sets that plus `V5_AZURE_VM=1`, `V5_ASSETS=/workspace/assets`, `V5_DEVICE=cuda:0`,
  `PYTHONPATH=/workspace/v5-perception/drone-flyby`, `$PY=/workspace/venv/bin/python`.

## Layout on the pod
- Code: `/workspace/v5-perception/drone-flyby` (dtos, utils, local_evaluator, v2/, v5/).
- Assets: `/workspace/assets` — `discovery.onnx` (final epoch-30, sha `66a77df…`), `dinov2_224.onnx`,
  `visual_head.npz`, `manifest.json` (hashes match Azure), `yolo11s.pt`, `dataset_v52/`, `dataset_da/`.
- Domain-adapted detector: `/workspace/assets_da/discovery.onnx` (experiment `da1`).
- Hosted diagnostics: `/workspace/hosted-v3` (50 captures). Results: `/workspace/results`, training `/workspace/training/da1`.

## Run the endpoint
```sh
source /workspace/env.sh
cd /workspace/v5-perception/drone-flyby
V5_CANDIDATES=64 $PY -m uvicorn v5.endpoint:app --host 0.0.0.0 --port 9053 --workers 1
```
Runpod-internal port 9053 (NOT the Azure production 9053). Public HTTPS is via Runpod Connect proxy
(retrieve from the Runpod dashboard); do not submit it to the competition without owner approval.

## Verified GPU numbers (RTX 4090)
- Discovery 8.7 ms; DINOv2 encode 32/64/128/256 crops = 56/122/265/507 ms; classify-32 131 ms.
- Endpoint (budget 64, GPU idle): HTTP p50 135 / p95 197 / max 202 ms; 0/50 frames over 333 ms; 0 contract violations.

## Tooling (this dir)
- `bench_gpu.py` — GPU model/latency benchmark.
- `endpoint_client.py` — official-contract validation + latency over hosted frames.
- `eval_detector.py` — per-target IoU (full + tiled) for a given assets dir.
- `discovery_final_audit.py`, `tiled_discovery_test.py` — discovery recall analysis.
- `composite_dataset.py`, `train_da.py` — domain-adaptation dataset + training.
- `audit.py`, `fp_audit.py`, `compare_discovery.py` — target/background diagnostics.
- `manual_targets.json` — human diagnostic boxes (approximate, dev-only, NOT GT).

## Known limitation
Detector hosted-target localization is not competition-grade: 0/8 manual targets reach IoU≥0.5
(both the epoch-30 Helsinki detector and the `da1` domain-adaptation run). Tiling lifts recall to
IoU~0.4 on most targets but not to 0.5; `t125_heli` is a complete miss. Requires better data
(masks/real backgrounds) or an owner-approved hosted Validation for a true score.
