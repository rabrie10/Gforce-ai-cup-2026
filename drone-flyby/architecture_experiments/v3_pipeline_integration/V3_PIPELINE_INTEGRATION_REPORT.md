# V3 Stage 2 — controlled full-pipeline integration

## Decision

Select **`v3_standard`, proposal threshold 0.01, NMS IoU 0.55, K=8** as the
single frozen candidate for an isolated VM benchmark.

The complete 25-frame Helsinki run improved from **0.4306 to 0.4591
mAP@0.50 (+0.0286)**. K=24 was nominally highest at 0.4605, but the extra
0.0014 mAP versus K=8 cost 2.7x as many Helsinki annotations, 2.3x the mean
track-bank size, and increased full-pipeline p95 from 431.5 ms to 616.1 ms.
K=8 is the controlled Pareto choice.

The local Windows CPU misses the 250–280 ms p95 target for both V2 and V3.
This is not an Azure decision. No deployment, live-VM change, hosted
Validation, or Final Evaluation was performed.

## Provenance and frozen assets

- Branch: `drone-v3-pipeline-integration`
- Integration starting point / selected V3 experiment commit:
  `74d1e0181d794b425f06d13a625df3de2fb6f7dd`
- Frozen V2 architecture reference:
  `6fb764e5b9fb5f7eefcb832c72b35553165219e5`
- V3 standard PT: `drone-flyby/models/v3_oneclass_standard.pt`, SHA-256
  `126aa31373775b324cf0df37c683b983202f9bc4dbfafd6105f40d98fafcec07`
- V3 standard ONNX: `drone-flyby/models/v3_oneclass_standard.onnx`, SHA-256
  `2daeabaeee41ffca9285485920238ce09c7762423d2d4fc41e406625d1b44d8f`
- The P2 model was not used.

Both hashes were reverified before every benchmark run. The production-path
copies are intentionally uncommitted model binaries. A deployment workspace
must transfer the two verified files from the preserved V3 experiment assets
to `drone-flyby/models/` before `docker build`. Docker copies the entire
`models/` directory and sets the candidate backend explicitly. Rollback is
`DRONE_DISCOVERY_BACKEND=v2`; V2 assets remain unchanged.

## Production integration

Only appearance discovery changed.

- `ProposalConfig.discovery_backend` explicitly selects `v2` or
  `v3_standard`; this integration branch defaults to `v3_standard`.
- V3 uses the existing class-agnostic YOLO ONNX decoder. Its one internal
  class (`target`) is discarded at discovery and never enters recognizer,
  track, or output class indices.
- Production startup verifies the frozen V3 asset SHA-256 and inference
  verifies the export has exactly one class before accepting proposals.
- Proposals are NMS/deduplicated, score ranked, hard bounded, and separately
  counted before and after K.
- ResNet18 recognition, gallery, affine GMC, track association, scheduler,
  output semantics, global geometry, and API schema are unchanged.
- Telemetry now includes discovery backend, before/after budget counts, score
  values, crop/background details, top class/posterior/objectness arrays,
  observations, bank creation/removal/saturation, emitted class/confidence,
  and per-stage/end-to-end latency. Session directories have caller identity
  plus a UUID; filenames retain frame, sequence, and request identity.
- Docker has explicit V3 paths and the one-variable V2 rollback.

## Helsinki: complete pipeline

All mAP numbers below use the unchanged repository COCO scorer on all 25
consecutive frames. The separate real-time run uses the same renderer, camera
rules, response validation, and 333 ms frame-clock skip logic.

| Backend | K | offline mAP@.50 | realtime mAP | accepted / skipped | ann./frame | classes | bank mean / max | total p50 / p95 / max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 32 | 0.4306 | 0.1107 | 8 / 17 | 22.52 | 15 | 9.80 / 15 | 369.3 / 442.7 / 446.8 |
| V3 standard | 8 | **0.4591** | **0.1259** | 8 / 17 | 20.80 | 16 | 9.56 / 12 | 374.3 / 431.5 / 521.5 |
| V3 standard | 16 | 0.4553 | 0.1228 | 7 / 18 | 41.76 | 16 | 16.92 / 20 | 464.6 / 540.4 / 568.7 |
| V3 standard | 24 | 0.4605 | 0.1222 | 7 / 18 | 57.00 | 16 | 22.16 / 29 | 508.9 / 616.1 / 637.5 |
| V3 standard | 32 | 0.4598 | 0.1238 | 7 / 18 | 60.56 | 16 | 23.20 / 32 | 526.3 / 630.7 / 678.5 |

No response or camera command was invalid. No run saturated the 48-track
hard ceiling. V3 K=8 created 37 and removed 25 tracks over the complete run,
versus 29 and 24 for V2. Higher K created 65–89 tracks and inflated output
without material AP gain.

### Per-class AP@0.50

| class | V2 K32 | V3 K8 | V3 K16 | V3 K24 | V3 K32 |
|---|---:|---:|---:|---:|---:|
| condor | 0.9010 | 0.9010 | 0.8605 | 0.7933 | 0.7824 |
| hangar | 0.6634 | 0 | 0 | 0 | 0 |
| helicopter | 0.9447 | 0.9029 | 0.8066 | 0.8094 | 0.8094 |
| jammer | 0 | 0.7624 | 1.0000 | 1.0000 | 1.0000 |
| jet_plane | 0.9193 | 0.9443 | 0.9170 | 0.9105 | 0.9079 |
| large_launcher | 0.9845 | 0.9895 | 0.9810 | 0.9694 | 0.9687 |
| large_tower | 0 | 0 | 0 | 0 | 0 |
| medium_launcher | 0 | 0 | 0 | 0 | 0 |
| medium_plane | 0 | 0 | 0 | 0 | 0 |
| mine_roller | 0 | 0 | 0 | 0 | 0 |
| small_launcher | 0 | 0 | 0 | 0 | 0 |
| small_plane | 0 | 0 | 0 | 0 | 0 |
| small_tower | 0.9352 | 0.9352 | 0.8188 | 0.9140 | 0.9171 |
| spacecraft | 0.9373 | 0.9505 | 0.9505 | 0.9505 | 0.9505 |
| ta-ta | 0 | 0 | 0 | 0.0775 | 0.0775 |
| tank | 0.6040 | 0.9604 | 0.9496 | 0.9434 | 0.9429 |

V3's discovery gain survives the full offline system at K=8, chiefly through
new jammer performance and a large tank gain, while losing hangar. The main
gain stops scaling at recognition/background rejection: additional discovery
candidates are mostly admitted as observations, expanding tracks and outputs
without improving macro AP.

## Hosted captures: GT-free diagnostics

The 50 images are stride-5 samples. Each image was processed independently
through discovery and recognition at L0; no tracker/GMC/retention metric was
calculated and no hidden recall, IoU, AP, or class accuracy is claimed.

| Backend | K | proposals before / after | rejection | observations/image | discovery+recognition p95 ms |
|---|---:|---:|---:|---:|---:|
| V2 | 32 | 0.82 / 0.82 | 2.7% | 0.76 | 237.9 |
| V3 standard | 8 | 54.72 / 8.00 | 21.8% | 6.26 | 314.2 |
| V3 standard | 16 | 54.72 / 16.00 | 22.0% | 12.48 | 406.7 |
| V3 standard | 24 | 54.72 / 24.00 | 22.0% | 18.72 | 503.4 |
| V3 standard | 32 | 54.72 / 31.82 | 22.9% | 24.54 | 564.1 |

The 54.72 figure is after existing NMS/geometry/deduplication and before K;
it is therefore lower than the earlier proposal-only 60.66 activation figure.
At K=8 the recognizer rejects only 21.8%, so 6.26 candidates/image enter as
observations. Jammer and ta-ta dominate the GT-free top-class distribution.
This is a false-positive risk indicator, not evidence those classes exist.

## Validation and readiness

- All 49 unit/integration tests pass, including targeted one-class discovery,
  hard budget, official-class-only output, global normalized geometry,
  request/frame echo, legal camera transitions, no L0→L2 jump, off-crop state,
  bounded tracks, corrupt input, and telemetry isolation.
- All benchmark responses and camera commands were valid; inference failures
  are contained by existing defensive fallbacks.
- The selected candidate emits all 16 official names over Helsinki and never
  emits `target`.
- Local K=8 proposal/recognition p95 was 210.8/155.2 ms and total p95 was
  431.5 ms. Timing is single-host Windows evidence, not an Azure VM result.
- Docker wiring is ready once the two hash-verified, uncommitted model files
  are transferred into `drone-flyby/models/`. The existing V2 image/assets and
  explicit backend switch preserve a straightforward rollback.

## Limitations and risks

- Helsinki has only 25 frames and includes the same physical instances used
  elsewhere in development; it does not establish broad generalization.
- Hosted captures have no ground truth and are non-consecutive stride-5
  samples. Their activation cannot establish accuracy.
- K=8 still admits far more hosted observations than V2, and the recognizer's
  modest rejection rate may create background tracks on a long sequence.
- Local full-pipeline latency is above both the 333 ms interval and requested
  safety margin. The isolated target-VM benchmark is therefore mandatory.
- Model binaries are not committed; packaging depends on explicit hash-checked
  transfer before building.

## Exactly one recommended next action

**Run an isolated Azure VM benchmark of the selected integrated V3 candidate.**
