# Architecture V2

The closed loop, and why each piece is shaped the way it is. Every design claim
below traces to a measurement in the evaluator probes, the dataset & scene
audit, the motion oracles, the Baseline 1 forensic audit, or the component
probes in `architecture_experiments/v2_component_probes/`.

```
L0 global discovery / motion anchor
    -> global track bank
    -> camera scheduler          <- built and tested; see finding 1 for why it
    -> L1 routine identity refresh     currently holds L0 rather than zooming
       (selective L2 disambiguation)
    -> transfer-oriented crop recognizer
    -> identity evidence fusion
    -> affine GMC / temporal state
    -> full-frame global outputs
    (repeat)
```

## Modules

| Module | Responsibility |
| --- | --- |
| `config.py` | Every knob, env-overridable. Components switch off without code changes. |
| `geometry.py` | view <-> source <-> global, boxes, affine algebra, level rendering. |
| `proposals.py` | Class-agnostic discovery: saliency band-pass and/or detector boxes. |
| `gallery.py` | Crop preprocessing, augmentation, prototype reduction, storage. |
| `embedders.py` | Interchangeable crop representations, for the pipeline and probe B. |
| `recognizer.py` | Frozen-backbone prototype matching, posterior + reject option. |
| `tracks.py` | Global state in source pixels: boxes, identity evidence, staleness. |
| `gmc.py` | Image-derived affine shared motion, with hard quality gates. |
| `scheduler.py` | Deterministic utility-driven camera policy. |
| `output.py` | Confidence fusion, top-k class emission, deduplication, DTO. |
| `telemetry.py` | Non-blocking validation capture. |
| `pipeline.py` | The loop, per-component timing, defensive degradation. |

`example.py` is a thin adapter; `api.py` still owns the transport.

## The three findings that shaped this

### 1. Active resolution is built, tested, and currently switched off - on evidence

The prior was that L1 should be the working level. The scene audit's L0
short-side histogram is `<4px: 10, 4-8: 50, 8-16: 121, 16-32: 67, >32: 11` out
of 259 appearances, and the Baseline 1 audit found **all 60 appearances under
8 px at L0 received no IoU>=0.50 localization** while 77/78 at >=16 px were
correct. At L1 only 10/259 fall under 8 px. That argument is sound, and it is
about *detection*.

It does not survive contact with the detector we actually have. Probe C
measured recall against apparent size **in the detector's own input**: 0.00
below 8 px, 0.41 at 8-16 px, 0.98 at 16-32 px. Those weights were trained on L0
renders only, so at L1 the view has to be downscaled by two to stay inside that
trained range — which hands back exactly the detail the zoom just bought.
Feeding L1 at native scale instead is worse still (0.200 vs 0.464 centre
recall), because the detector has no prior for objects twice the size it ever
saw.

End to end on Helsinki, offline: **`working_level=0` scores 0.425,
`working_level=1` scores 0.341.** L1 costs 75% of per-frame coverage and
returns only the recognizer's small detail gain.

So L0 is the working level **for now**, and the scheduler, the track bank and
the legal-transition machinery all exist and are tested behind one environment
variable. The unlock is a proposal detector trained across L0/L1/L2 renders
with scale augmentation, so that discovery recall improves with resolution
instead of being invariant to it. That is the highest-value open item here.

### 2. Within-flight recognition evidence is confounded, and the confound is the failure mode

Probe A rendered every GT appearance at true L0/L1/L2 and classified it by
nearest-exemplar cosine against temporally distant references. Raw 16x16 grey
pixels scored **0.927 top-1 at L0**, on objects two to fifteen pixels across.

That is impossible as identity, so the probe added a control: erase the object
and keep only its surroundings.

| variant | pixels | resnet18 | resnet50 | yolo_head |
| --- | --- | --- | --- | --- |
| full (L1) | 0.931 | 0.923 | 0.934 | 0.533 |
| object_only (L1) | 0.857 | 0.911 | 0.927 | 0.486 |
| **background_only (L1)** | **0.938** | **0.911** | **0.931** | **0.556** |

**The background alone answers the question.** The objects are static on the
ground and only the camera moves, so every crop of a given object contains the
same patch of terrain for its whole life. This upgrades the Baseline 1 audit's
"background/context memorization: PLAUSIBLE, unproven" to a measured mechanism,
and it is a direct candidate explanation for 0.3837 local against 0.0016 hosted:
the terrain does not come with us to a new flight.

Three consequences are built into this code:

* **Tight crops** (context 1.25, not 1.6).
* **Background normalization** - the crop's border-ring median is subtracted
  before embedding, so identity depends on the object's contrast structure and
  not on the absolute colour of the ground.
* **Rotation augmentation** - the reference flight fixes every asset's heading
  relative to the camera; another sequence will not.

The honest residual: even `object_only` is one instance per class, so **no
number from this dataset is cross-instance generalization evidence.** What the
probe does support is the relative reading - L2 buys nothing over L1 for
identity (resnet18 object_only: L0 0.880, L1 0.911, L2 0.911), which is why L2
is a selective dwell rather than the working level.

### 3. Discovery was the bottleneck, and it needed a reject option

The first integrated run scored 0.082 offline. The offline diagnostic attributed
it precisely:

* recognizer top-1 on proposals that hit a real object: **26/28 = 0.93**
* proposal centre recall: **0.341**
* tracks per frame: **100.5**, of which on a real object: **2.44**
* **track precision: 2.4%**, and 177.6 annotations emitted per frame

The recognizer was not the problem. The problem was that it had **no way to say
"not an object"**: a softmax over sixteen class prototypes names every grass
blob something. So the gallery now carries background prototypes mined from the
proposal engine's *own* false positives, and `objectness` - best class prototype
against best background prototype - gates admission to the track bank.

## Component decisions

| Component | Chosen | Rejected, and why |
| --- | --- | --- |
| Crop representation | frozen resnet18, 512-d | `yolo_head` (Baseline 1's supervised head) scored 0.47-0.53 against 0.88-0.93 on **identical crops**: strictly less signal, not complementary. resnet50 is ~equal and ~2x the CPU. |
| Identity | prototype cosine + reject | 16-way fine-tuning: measured to transfer at 0.0016. |
| Working level | **L0, for now** | L1 measured 0.341 against L0's 0.425 end to end: discovery recall is invariant to level with the current detector (probe C), so the zoom costs coverage and buys nothing. L2 adds no identity over L1 either (probe A). |
| GMC | affine, similarity then median-translation fallback | Homography: D-007 stands, no new measured problem justifies reopening it. |
| Residual | none | D-007's caveat: matched fixed-size affine beat affine+frozen residual in the oracle. |
| Camera policy | deterministic utility | RL/POMDP: out of scope for this pass. |

## Configuration

Everything is `DRONE_*` environment variables (see `config.py`). The ones worth
knowing:

```
DRONE_SCHED_LEVEL=1        # turn active resolution on (measured worse today)
DRONE_RECOGNIZER=0         # uniform posteriors; isolates discovery
DRONE_GMC=0                # per-track constant velocity only
DRONE_TELEMETRY=1          # validation capture on
DRONE_OUT_TOPK=1           # emit only the best class per track
```

## Why top-k class emission

COCO AP is computed **per class independently**, so a box labelled `tank` can
only ever cost `tank`'s precision and can never touch `condor`. Baseline 1
emitted 7 of 16 class IDs and was therefore capped at 0.44 before any other
error. Each track names its top 3 classes at damped confidence: a correctly
ranked second guess turns a guaranteed zero into a real AP, and within-class
ranking is what AP integrates over.

## Measured state

End to end on Helsinki, offline, `local_evaluator.py`:

| | |
| --- | --- |
| mAP@0.50 | **0.4310** |
| frames accepted | 25/25, 0 invalid, 0 timeouts |
| classes with nonzero AP | 8/16 |
| evaluator oracle | 1.000 |

Per-component latency, dev host, four torch threads and four ORT threads:

| stage | mean | p95 |
| --- | --- | --- |
| proposals (ONNX detector) | 144-227 ms | 211-276 ms |
| recognize (resnet18 + gallery) | 89-136 ms | 138-204 ms |
| decode | 37-59 ms | 63-75 ms |
| gmc | 17-25 ms | 24-27 ms |
| schedule + emit + update | < 5 ms | < 10 ms |
| **total** | **402 ms** | **~480 ms** |

The total is from a paired A/B (three repetitions, telemetry off: 402.4 /
403.4 / 401.9 ms). Individual runs on this host ranged 290-453 ms depending on
what else was running, which is why the controlled figure is quoted.

**This is at or above the 333 ms frame interval, and it is the top deployment
risk.** A frame that arrives late is skipped and scored as no detections, so
latency converts directly into lost recall. VM latency on the four-vCPU
`B4as_v2` target is **not measured**. Levers, cheapest first:
`DRONE_PROPOSAL_YOLO_IMGSZ` (proposals are ~50% of the frame),
`DRONE_PROPOSAL_BUDGET`, `DRONE_RECOGNIZER_CROP`.

Local realtime scores 0.072-0.097 with 19-20 of 25 frames skipped, but the
local harness loads, crops, re-encodes and base64s a 4K PNG inside its own
timing loop on the same four cores. Baseline 1 saw the same effect at a 240 ms
endpoint. Treat local realtime as a lower bound, not a prediction.

## Running it

```bash
python tools/build_gallery.py           # offline, once; writes models/reference_gallery.npz
python api.py                           # serves /predict, /metrics, /api
python local_evaluator.py               # offline score
python local_evaluator.py --realtime    # score under the 3 fps clock
python tools/diagnose_offline.py        # attribute a score to a stage
python -m unittest discover -s tests
```

`GET /metrics` reports per-component latency and what is actually loaded.
