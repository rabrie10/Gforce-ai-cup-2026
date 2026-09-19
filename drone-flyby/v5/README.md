# V5.2 perception implementation

Independent pretrained YOLO11s discovery and frozen DINOv2 ViT-S/14 recognition,
with a regularized direct 16-class head, reference similarity, binary target
assessment, uncertainty, causal global tracks and optional legal camera control.

All model execution is explicitly gated to `V5_AZURE_VM=1` on Linux. Development
and source control take place in the isolated Windows worktree. Production
`api.py`, `example.py`, the official DTOs/evaluator, and V2/V3 assets are unchanged.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `V5_ASSETS` | `/assets` | Directory with hash-verified model manifest |
| `V5_DEVICE` | `cpu` | `cpu` or `cuda:0`; CUDA requires GPU ONNX Runtime and driver |
| `V5_THREADS` | `2` | CPU inference threads per model |
| `V5_CANDIDATES` | `32` | Ranked recognition budget, independently configurable |
| `V5_HEAD` | `visual_head.npz` | Head fixes the matching encoder input size |
| `V5_TEMPORAL` | `1` | Causal tracking enabled |
| `V5_ACTIVE_CAMERA` | `0` | Explicit L0-only default; `1` enables native L1/L2 requests |
| `V5_MULTISCALE` | `0` | Add four spatial detector tiles; costs four extra passes |

The primary head uses **224-pixel crops**, retaining YOLO11s and DINOv2. The
98/140 exports are measured configuration alternatives, not replacement models.
The full 32-crop configuration exceeds the current CPU's 333 ms throughput budget.
Do not silently select a smaller model, crop size or candidate budget to hide this.

## Data and validation

Source: supplied original Helsinki 3840x2160 PNGs and genuine annotation JSON.
Training frames: 0–20 excluding calibration frames 4, 9, 14, 18. Test frames:
21–24. **Hangar and medium_plane are absent from frames 0–18**, so the original
split could not support a complete 16-class supervised head. The class-coverage
guard caught this before publishing a head.

Each source produces a full L0 view and up to four object-centered legal L1 and
four L2 views. Every intersecting annotated box is transformed and clipped. Two
photographic background crops per source have no overlap with any supplied target
box plus a 32-source-pixel guard. These negatives are resized background samples,
not claimed native camera observations. No hosted image is assumed target-free.
No pasted object rectangles, invented masks, detector labels, or V3 predictions
are used as supervision. No trustworthy segmentation assets were found, so
object compositing is omitted.

Visual training uses genuine camera-rendered object crops with modest photometric
variation and in-plane rotations/flips. These do not establish 3D viewpoint
coverage. The encoder stays frozen. Direct multinomial logistic regression has
L2 regularization (`C=10`); the separate binary head uses `C=3`. Reference matching
uses unaugmented training crops; its distribution is independently reported.
The final class distribution is 80% direct head plus 20% reference distribution.

The object-erased control replaces the complete annotation rectangle with border
median color while retaining the surrounding crop. It is an ablation rather than
a claimed object segmentation. It can reveal background dependence; low scores
would still not prove new-instance generalization. Helsinki has one physical
instance per class. All frame-held-out results concern those same instances.

`train_visual.py` reports oracle-box top-1/top-3, direct/reference comparisons,
confusion matrices and background acceptance **without discovery**.
`evaluate.py` reports raw and ranked discovery recall, localization, unmatched
proposals and photographic-negative false positives. It imports the unchanged
official scorer for integrated evaluation and calls its HTTP replay directly.

## Reproduction on Azure

Build `v5/Dockerfile` in an isolated Azure environment using the frozen V3 runtime
image as dependency base. The V3 model weights in that image are never loaded by
V5. Bind the V5 checkout at `/work`, writable V5 assets at `/assets`, V5 results at
`/results`, and V5 cache at `/cache`. No existing service needs restarting.

Run from `/work`:

```sh
python -m v5.feasibility
python -m v5.export_encoder --sizes 98 140 224 --batches 8 16 32 64
python -m v5.data
python -m v5.train_visual --size 224
python -m v5.train_discovery --epochs 30
python -m v5.package
python -m unittest discover -s v5/tests -v
python -m v5.replay --inputs /hosted-v3 --out /results/hosted
python -m uvicorn v5.endpoint:app --host 0.0.0.0 --port 9353 --workers 1
python -m v5.evaluate --url http://127.0.0.1:9353/predict
```

The isolated endpoint must be published only to `127.0.0.1:9353` after inspecting
active ports. Its `/reset` route starts an isolated test episode; `/metrics`
reports in-process endpoint latency. `evaluate.py` also measures complete HTTP
round-trip timing and the official 333 ms frame-clock skipped fraction. The
training script can run visual feature extraction and detector training together
under an aggregate resource cap. Do not benchmark while training is competing.

For CUDA, install `onnxruntime-gpu` instead of `onnxruntime` in a separate NVIDIA
CUDA runtime, use `--gpus all`, and set `V5_DEVICE=cuda:0`. CUDA selection fails
explicitly if its execution provider is unavailable. GPU performance is unverified
until the identical model hashes and requests are measured there. No paid resource
is provisioned by these scripts.

## Hosted inspection

`replay.py` reads exact paired recorded images and V3 JSON from a supplied capture
directory. Capture index and source frame are shown separately (index 60 may be
source frame 61). Five development captures (indices 0,60,125,185,245) are selected;
other saved captures remain untouched by the V5 development replay. No hosted AP,
recall or class accuracy is claimed without verified boxes/classes.

Outputs include raw images, saved V3 predictions, V5 raw/ranked proposals, final
predictions, full per-candidate JSON and enlarged quadrants for visual review.
The raw overlay caps drawing at 300 proposals for readability; JSON preserves
the full above-threshold pre-NMS set. Class scores are absent on candidates not
passed to the expert, and the budget filtering reason is explicit. They must not
be invented. Manual-box panel is explicitly unavailable when no exact verified
machine-readable annotations can be matched to that image.

## Temporal and camera status

Only past/current images feed source-space affine motion and association. Missing
off-crop tracks require accepted motion and expire after three source frame
indices. In-view missing tracks are suppressed. Nearly identical embeddings at
the same/lower resolution do not increase identity evidence. A bounded EMA permits
later class reversal. Class alternatives stay internal; unknown is never emitted.
Same-class overlapping output boxes are suppressed. Up to four sequence states
and eight exact request responses per sequence are retained, with out-of-order
requests safely empty rather than answered using future state.

Active camera control is optional and uses the existing legal movement planner.
It considers track age, uncertainty, recent zoom use, lost scene coverage and
periodic L0 refresh; L2 returns through legal L1 transitions. No scheduler training
or policy claims are made. End-to-end camera benefit still needs measurement.
No auxiliary family head or pair specialist is enabled without measured confusion
supporting it; direct 16-class inference is always available.
