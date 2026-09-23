# Architecture V2 component probes

These are not standalone studies. Each one exists to choose or reject a concrete
V2 component, and each was run **during** the V2 build rather than before it
(D-008). Raw machine-readable results are in the `*_results.json` files beside
each script.

| Probe | Question | Outcome |
| --- | --- | --- |
| A + B | `exp_a_crop_resolution.py` - does true camera resolution restore identity, and which representation carries it? | Question A is **not answerable on this dataset**; question B decided the representation. |
| C | `exp_c_proposal_recall.py` - class-agnostic proposal recall under a bounded budget. | See below. |
| D | `exp_d_affine_feasibility.py` - can real pixels estimate the affine the GT oracle selected? | See below. |
| — | `../../tools/diagnose_offline.py` - attribute an end-to-end score to a stage. | Found the real bottleneck. |

---

## Probe A + B — oracle crop recognition at true L0/L1/L2

**Protocol.** Every GT box places (never answers) a crop. Each appearance is
rendered through the exact transmission path the evaluator uses — source region,
INTER_AREA downsample by the level factor, resize to 64x64 — so the crop carries
the detail the level would really have carried. Identity is nearest-exemplar
cosine over a gallery of other appearances, with a **temporal exclusion band**
of 4 frames so the adjacent near-duplicate frame cannot act as a reference.
259 appearances, 16 classes, 25 frames.

**The first run invalidated its own protocol.** Raw 16x16 grey pixels scored
0.927 top-1 at L0, on objects two to fifteen pixels across. So a control was
added: erase the object, keep the surroundings.

Top-1 accuracy, `full` / `object_only` / `background_only`:

| representation | L0 | L1 | L2 |
| --- | --- | --- | --- |
| pixels (16x16 grey) | 0.927 / 0.857 / 0.923 | 0.931 / 0.857 / 0.938 | 0.931 / 0.857 / 0.946 |
| resnet18 (frozen) | 0.927 / 0.880 / 0.919 | 0.923 / 0.911 / 0.911 | 0.927 / 0.911 / 0.915 |
| resnet50 (frozen) | 0.938 / 0.934 / 0.915 | 0.934 / 0.927 / 0.931 | 0.919 / 0.927 / 0.927 |
| yolo_head (Baseline 1) | 0.494 / 0.467 / 0.533 | 0.533 / 0.486 / 0.556 | 0.486 / 0.506 / 0.560 |

`object_only` fills everything outside the GT box with a **constant neutral
grey**. An earlier revision filled it with the border mean and leaked the
terrain colour — the very thing the variant exists to remove.

### Findings

**A1 — Terrain context alone solves the task (0.911–0.946).** The objects are
static on the ground and only the camera moves, so each object keeps the same
patch of terrain for its whole life. This **upgrades the Baseline 1 audit's
"background/context memorization: PLAUSIBLE, unproven" to a measured mechanism**,
and is a direct candidate explanation for 0.3837 local versus 0.0016 hosted.

**A2 — Question A cannot be answered with this dataset.** Any within-flight
nearest-neighbour protocol can cheat through terrain. The only resolution effect
that survives the strongest available control is resnet18 `object_only`:
L0 0.880 → L1 0.911 → **L2 0.911**. Small, and flat from L1 to L2.

**A3 — L2 buys no identity over L1.** Consistent across resnet18 (0.911 →
0.911) and resnet50 (0.927 → 0.927). Combined with the scene audit's coverage
figures (median 5 complete objects per L1 crop against 3 at L2), this rules L2
out as a working level. Probe C then ruled out L1 as well, for a different
reason — see C4.

**A4 — The supervised branch is rejected.** On **identical crops**, Baseline 1's
own head scores 0.47–0.53 where a frozen ImageNet resnet scores 0.88–0.93. It
carries strictly less usable crop identity signal, so there is no complementary
signal to fuse. This is exactly the controlled identical-crop comparison the
handoff required before allowing a supervised branch, and it fails it.

**A5 — resnet18 over resnet50.** Equivalent on `object_only` (0.911 vs 0.927)
at roughly half the CPU (8.4 vs 15.0 ms/crop measured).

### What this does NOT show

One physical instance per class means even `object_only` is instance matching.
**No number here is cross-instance or cross-sequence generalization evidence**
and none may be quoted as such. Nothing in this probe justifies a resolution
level on its own; the level decision rests on detection evidence (probe C and
the Baseline 1 audit) and on the end-to-end comparison.

### Consequences built into V2

* tight crops (context 1.25, not 1.6);
* border-ring median subtraction before embedding, so identity depends on the
  object's contrast structure rather than the absolute colour of the ground;
* rotation augmentation of the gallery, because the reference flight fixes each
  asset's heading relative to the camera and another sequence will not.

---

## Probe C — class-agnostic proposal recall under a bounded budget

**Protocol.** Each backend runs on the rendered view at each level. Recall is
computed only over GT objects lying completely inside that view, two ways:
`iou50` (a proposal matches the GT box at IoU >= 0.50) and `center` (a proposal
merely contains the GT centre). The second is the criterion the *scheduler*
actually needs — on a four-pixel object, IoU 0.50 is a box-regression test, not
a discovery test.

### First run (original settings, 80-proposal budget)

| backend | level | mean proposals | ctr recall @80 | iou50 @80 |
| --- | --- | --- | --- | --- |
| saliency | L0 | 79.9 | **0.046** | 0.000 |
| saliency | L1 | 79.9 | **0.024** | 0.000 |
| yolo | L0 | 6.4 | 0.467 | 0.463 |
| yolo | L1 | 1.3 | **0.200** | 0.156 |
| hybrid | L0 | 80.0 | 0.467 | 0.463 |
| hybrid | L1 | 80.0 | 0.216 | 0.156 |

**C1 — The saliency proposer is rejected.** 0.046 and 0.024 centre recall with
eighty proposals: it ranks terrain texture above man-made objects, and adds
essentially nothing to the union (hybrid ≈ yolo). It stays in the tree, off by
default, documented as measured-and-rejected — a scale-space blob detector with
a proper significance test is still a reasonable idea, but this is not one.

**C2 — The detector's L1 collapse has a mechanical cause.** 0.467 at L0 against
0.200 at L1. The weights were trained on L0 renders, so at L1 every object is
twice the apparent size the detector ever saw. Feeding it the received image at
`960 >> level` restores the trained apparent size — and costs proportionally
less CPU at exactly the level the scheduler spends most of its time at.

**C3 — Per-frame recall is not the binding constraint it looks like.** 0.467 is
low, but an object only has to be discovered *once* to enter the track bank and
then be carried by GMC. Over a 249-frame sequence, per-frame discovery
probability compounds; this is the main reason the bank exists.

### Second run (level-matched input, `960 >> level`)

| level | mean proposals | ctr recall @20 | iou50 @20 |
| --- | --- | --- | --- |
| L0 | 8.8 | **0.490** | 0.486 |
| L1 | 1.9 | **0.464** (was 0.200) | 0.464 |
| L2 | 0.8 | 0.407 | 0.407 |

Level-matching more than doubled L1 recall. But the size breakdown shows what
it really did, and the answer matters more than the headline:

| apparent short side **in the detector's own input** | ctr recall |
| --- | --- |
| `<4 px` | 0.000 |
| `4-<8 px` | 0.000 |
| `8-<16 px` | ~0.41 |
| `16-<32 px` | ~0.98 |
| `>=32 px` | ~0.95-1.00 |

At L0 the `8-<16` bucket recalls 0.413 and `16-<32` recalls 0.985. At L1 the
corresponding buckets — one level up, so `16-<32` and `>=32` in L1 apparent
terms — recall 0.385 and 0.947. **Identical curve.** Recall depends only on
apparent size in the detector's input, and level-matching puts L1 back exactly
where L0 was.

**C4 — the active-resolution premise is currently blocked on discovery.**
Level-matching is the better of the two options measured (0.464 vs 0.200), but
it necessarily hands back the detail the zoom just bought. With these weights,
L1 cannot see anything L0 cannot. That prediction was then confirmed end to
end: `working_level=0` scores **0.425** and `working_level=1` scores **0.341**.

The unlock is a **proposal detector trained across L0/L1/L2 renders with scale
augmentation**, so that discovery recall improves with resolution instead of
being invariant to it. Until then, L1 costs 75% of per-frame coverage and
returns only the recognizer's small detail gain.

**Caveat.** Latencies in these tables were measured with another job running
and are not clean. Component latency is reported separately from `/metrics`.

---

## Offline diagnosis — where the integrated score actually went

The first fully integrated run scored **0.082** offline. `diagnose_offline.py`
attributed it:

| quantity | value |
| --- | --- |
| recognizer top-1 on proposals that hit a real object | **26/28 = 0.929** |
| proposal centre recall | **0.341** |
| tracks per frame | **100.5** |
| tracks on a real object per frame | **2.44** |
| **track precision** | **2.4%** |
| annotations emitted per frame | **177.6** |

The recognizer was not the bottleneck. The bank was 97.6% junk because the
recognizer had **no reject option**: a softmax over sixteen class prototypes
names every grass blob something. The fix is background prototypes mined from
the proposal engine's *own* false positives, plus an `objectness` gate
(best class prototype vs best background prototype) applied before anything
reaches the bank.

A second measured problem from the same run: the saliency proposer used
morphological top-hat with up to a 63x63 structuring element, costing
**1690 ms per frame** — five times the entire 333 ms frame interval. Replaced
with a separable box-filter band-pass, which is O(1) in the window size.

After the reject option and the discovery fix, the same diagnostic is healthy:
`emit` 2.3 ms and `schedule` 0.2 ms per frame, against 44.7 ms and 68.8 ms when
the bank held a hundred tracks.

---

## Component ablations — end to end on Helsinki, offline

Each row changes exactly one environment variable against the shipped default.

| configuration | mAP@0.50 | endpoint mean (ms) |
| --- | --- | --- |
| **shipped default** (L0, recognizer on, GMC affine, top-3) | **0.425** | 362.8 |
| `DRONE_SCHED_LEVEL=1` (active resolution) | 0.341 | 361 |
| `DRONE_GMC=0` | 0.409 | 355.8 |
| `DRONE_OUT_TOPK=1` | 0.426 | 392.0 |
| `DRONE_OUT_TOPK=5` | 0.425 | 383.6 |
| `DRONE_RECOGNIZER=0` | **0.017** | 283.5 |

**The recognizer carries the system.** Removing it costs 0.408 of 0.425. This
is the component the hosted attempt is really testing.

**GMC is worth +0.016 here, and should be worth more hosted.** The camera is
static at L0 in this configuration, so almost every track is refreshed by a
direct observation every frame and there is little for propagation to do. Its
value is bridging gaps — which is exactly what skipped frames and any future
camera movement create.

**top-k is measured neutral, and is kept as a hedge, not as a gain.** 0.426 /
0.425 / 0.425 across k=1/3/5 is noise. It cannot show a benefit here because
local identity is nearly always correct; its justification is structural —
macro-AP over 16 classes, Baseline 1 capped at 0.44 by emitting only 7 of them,
and a measured inability to predict identity quality on an unseen landscape.
One environment variable reverts it.

---

## Probe D — can real pixels estimate the affine the GT oracle selected?

D-007 chose affine on GT-oracle evidence (strict LOO, 238/243 = 0.9794 at h=1).
This probe asks the production question with the same yardstick: push a GT box
through **image-derived** transforms for h steps and check IoU >= 0.50 against
the GT box h frames later. A rejected transform counts as an eligible failure,
so these rates include the cost of quality gating.

Affine, by camera policy:

| policy | accepted | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | median centre err h=1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | 24/24 | **0.9588** | 0.9471 | 0.9434 | 0.9340 | 0.9341 | **0.9286** | **0.79 px** |
| L1 static | 24/24 | 0.9588 | 0.9471 | 0.9340 | 0.9239 | 0.8956 | 0.8869 | 1.03 px |
| L1 sweeping | 24/24 | 0.9588 | 0.9427 | 0.9387 | 0.9188 | 0.9176 | 0.9226 | 1.29 px |

Similarity, same protocol:

| policy | h=1 | h=3 | h=6 |
| --- | --- | --- | --- |
| L0 | 0.9424 | 0.8726 | 0.6429 |
| L1 static | 0.9465 | 0.8349 | 0.5833 |
| L1 sweeping | 0.9465 | 0.8019 | 0.5060 |

### Findings

**D1 — image-derived affine essentially matches the GT oracle.** 0.9588 at h=1
against the oracle's 0.9794, with 0.79 px median centre error against the
oracle's 0.932 px non-boundary. At h=6 it holds **0.9286**, against the
oracle's best fixed-size variant at 147/155 = 0.948 and its residual variant at
135/155 = 0.871. **D-007 is confirmed with real pixels, not just GT.**

**D2 — affine beats similarity decisively, and the gap widens with horizon.**
0.9286 against 0.6429 at h=6. This is the same ordering the oracle found, which
is the result that mattered: the transform-class choice survives the move from
ground truth to correspondences.

**D3 — 100% transform acceptance (24/24) under every policy**, including a
camera moving 960 px per frame, where the inlier ratio falls to 0.66 and the
estimate still passes every quality gate. Latency 20-23 ms.

**D4 — propagation is not the limiting factor.** Tracks can be carried six
frames with >92% box validity. Whatever is limiting this architecture, it is
not the ability to remember where things are — which is why discovery and
latency, not state, are where the remaining work is.

One implementation note earned the hard way: the optical-flow pyramid must span
the largest displacement the canvas can show. At the default canvas width the
estimate initially failed outright for large motions, and a pyramid too shallow
does not degrade gracefully, it simply returns nothing.
