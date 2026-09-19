# V4 active-resolution: Phase A gate

Status: prepared, **not empirically evaluated**. No Phase B scheduler is implemented;
no evidence yet permits it. No production files, model assets, endpoint, or V3
defaults are changed. The isolated branch starts at
`82ce749fe7c471a0ef60a9115afca10dd1445121`.

## What this measures

`phase_a.py` requires Linux plus explicit `--azure-vm`. It reads all 25 original
3840x2160 Helsinki frames and annotations, renders through the supplied
`local_evaluator.Camera` / `render_view`, and decodes the actual transmitted PNG.
For every same-frame GT appearance it creates **oracle_target_centered** L0/L1/L2
views with legal static centers. These independent oracle crops are NOT a legal
camera trajectory, NOT a deployable selection policy, and NOT a coverage result.
They answer the decisive upper-bound question: can this frozen detector exploit
detail even when the camera has been perfectly aimed? GT never reaches recognition
or proposal inference. No archived hosted PNGs are used.

Each identical received crop is processed by the frozen ONNX proposer with:

* matched: requested longest side 960, 480, 240 for L0/L1/L2;
* native: requested longest side 960 at every level, without enlarging L0.

The native arm passes level=0 **only to the offline proposer call**. Recognition
still receives the real level and real source region. Serving code is untouched.
ONNX stride padding makes requested 240 a 256-wide canvas; both requested and
actual canvas dimensions are recorded. Model checksum validation remains enabled.
Model/recognizer errors abort the experiment; no saliency or uniform fallback may
produce apparently valid measurements.

K8 and K32 use prefixes of the same merged K300 pool. Discovery runs once per
mode/view and its time is shared across budgets; recognition is measured separately
for each budget. Duplicate oracle centers are cached within a frame so they do
not duplicate latency samples. A/B execution order alternates by frame. Warmup
is excluded. Latencies include proposal preprocessing/NMS and recognition crop
processing; PNG rendering, model loading and HTTP are excluded. This is a single
CPU pass, not an endpoint performance benchmark.

Outputs include raw global source boxes and confidences, per-object best IoU,
recall at IoU >= .50, mean best IoU including misses, source short-side buckets
<32 / 32–64 / >=64 pixels (equivalent L0 <8 / 8–16 / >=16), proposal burden at
IoU <.1 and <.5 against every frame GT, recognition rejection counts and stage
p50/p95 latency. Burden is per oracle crop, not representative whole-frame
precision. IoU <.5 includes localization failures and is not necessarily terrain.
An object appearance is identified by frame and annotation index, not class;
there is no one-instance-per-class assumption.

## Predeclared STOP / GO

For at least one of L1/L2, on >=30 paired object appearances at K8:

1. Native recall50 exceeds both matched same-level recall and L0 recall by >=10
   percentage points.
2. Native mean best IoU exceeds both comparators by >=0.05.

`summary.json` computes this gate. K32 and size buckets diagnose ranking versus
scale failures; K32 gains alone cannot pass. The threshold is an engineering
screen, not a significance test: appearances repeat across frames. GO authorizes
building and testing Phase B, not deployment. Inspect burden and latency before
choosing the useful level. STOP means stop scheduler development and consider a
targeted Research V2 question about scale coverage/ranking in the frozen proposer.
A failed run or absent summary means **NO DECISION**, never GO or detector failure.

## Exact Azure commands — Phase A first

Transfer this committed branch to the Azure repository using your normal Git
remote or a Git bundle. Run the following **on Azure**, from the existing
`Gforce-ai-cup-2026` repository. The branch must already exist locally. Models and
sample data remain read-only in the existing checkout; no copies overwrite them.
These commands only inspect the running 9053 container to obtain its immutable
runtime image ID. They do not exec into, restart, or replace it.

```bash
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
V4="$HOME/drone-v4-active-resolution"
test ! -e "$V4"
git worktree add --detach "$V4" drone-v4-active-resolution
git -C "$V4" merge-base --is-ancestor 82ce749fe7c471a0ef60a9115afca10dd1445121 HEAD
git -C "$V4" rev-parse HEAD
PROD_IDS=$(docker ps --filter publish=9053 --format '{{.ID}}')
test "$(printf '%s\n' "$PROD_IDS" | wc -l)" -eq 1
test -n "$PROD_IDS"
BASE_IMAGE=$(docker inspect --format '{{.Image}}' "$PROD_IDS")
ASSETS="$REPO/drone-flyby/models"
SCENE="$REPO/drone-flyby/src"
test -f "$ASSETS/v3_oneclass_standard.onnx"
test -f "$ASSETS/v3_oneclass_standard.pt"
test -f "$ASSETS/reference_gallery.npz"
test -f "$SCENE/helsinki/images/frame_000024.png"
RUN="$HOME/drone-v4-results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
IMAGE="drone-v4-probe:$(git -C "$V4" rev-parse --short=12 HEAD)"
docker build --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f "$V4/drone-flyby/architecture_experiments/v4_active_resolution/Dockerfile" \
  -t "$IMAGE" "$V4/drone-flyby"
docker image inspect "$IMAGE" > "$RUN/image.json"
git -C "$V4" rev-parse HEAD > "$RUN/commit.txt"
docker run --rm --network none --cpus 4 --memory 8g --read-only \
  --tmpfs /tmp:rw,size=256m \
  -v "$ASSETS:/experiment/models:ro" \
  -v "$SCENE:/experiment/src:ro" \
  -v "$RUN:/results:rw" \
  "$IMAGE" --azure-vm --out /results/phase-a 2>&1 | tee "$RUN/phase-a.log"
cat "$RUN/phase-a/summary.json"
```

Phase A uses **no published port** and no network. Reserve `127.0.0.1:9353:9053`
for the later justified candidate endpoint; never use 9053, 9153, or 9253 on the
host. Do not run simultaneous production traffic and CPU benchmarking if timing
is to be comparable; leave production running and schedule the probe while idle.
The base image provides the installed frozen stack and `/app/.torch` baked
ResNet18 weights. Network isolation deliberately makes missing assets fail.
Inherited DRONE_* configuration is cleared before imports and recorded by name;
all resolved config, pip versions, input/model hashes are saved in the manifest.

## Conditional next stage (not yet implemented or run)

If Azure reports GO, implement only a selectable native-zoom proposal knob and a
direction-independent policy using current proposals/tracks, legal transitions,
and periodic L0 refresh. The existing scheduler has a top-entry/bottom-exit prior;
simply changing DRONE_SCHED_LEVEL=1 would retain that prior and the resolution bug.
Keep full-frame track propagation and defensive response handling.

Then build the Phase C replay using the existing `run_integration.py` as a base:
fresh pipeline/camera per arm, explicit V3 K8 L0 vs gated V4 K8, identical 25 frame
IDs, supplied `build_request`, `Camera.apply`, `validate_response`, global source
box conversion and `local_evaluator.score` (COCO mAP50). Record received-level
counts BEFORE applying next-frame commands, accepted/refused command counts,
invalid frames, per-class AP, all-frame mAP50, and latency. Never score only visible
GT. Only attempt realtime after offline mAP exceeds baseline with zero invalid
frames/commands. No hosted Validation or Final Evaluation is authorized here.

## Local verification

`python -m unittest discover -s architecture_experiments/v4_active_resolution -v`
from drone-flyby runs dependency-free coordinate/metric/gate tests, no inference.
The Windows guard can also be checked by invoking phase_a.py; it must reject
before importing ML packages. Azure execution remains required to verify actual
model/data compatibility and establish STOP/GO. No V4 score improvement is claimed.

Prepared on Windows: five tests passed (including the subprocess inference guard),
Python compilation passed, and `git diff --check` passed. The bundled Python lacks
OpenCV, so real renderer integration was not locally executed; no packages were
installed and no model inference was attempted. This is an execution dependency
check pending Azure, not evidence that the detector cannot exploit zoom.
