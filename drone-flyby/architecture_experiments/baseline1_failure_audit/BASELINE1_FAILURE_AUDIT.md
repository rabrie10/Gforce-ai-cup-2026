# Baseline 1 Forensic Failure Audit

Audit date: 2026-09-19  
Frozen source commit: `129564f855060ea373dc8421701f97631162a828`  
Reviewed affine predecessor: `0ffe65733aef68f5c0109714e48de88ac2cbbe84`

## Executive answer

Baseline 1 taught us that a YOLO11n fine-tuned on 25 adjacent views of one
physical instance per class can fit a **subset** of the Helsinki reference
sequence, but the learned evidence does not transfer to the hosted scene. On
the saved Helsinki predictions it correctly detects 116/259 GT appearances
(44.8%) and emits only 7 of the 16 class IDs. It is excellent and temporally
stable on several repeated reference instances, completely silent on nine
classes, and catastrophically fails externally: local same-reference mAP@0.50
is 0.3837155 while hosted raw score is 0.00161904.

This is not primarily a box-regression failure. When the detector recognizes a
reference object, localization is usually strong. The dominant local failures
are **no detection** (131/259 GT), severe dependence on L0 pixel size, and a
smaller but clear identity-confusion mode (10 well-localized wrong classes).
The hosted collapse supports reference-instance/scene overfit, but without
hosted GT it is impossible to identify which hosted classes, sizes,
localizations, or domain factors caused that collapse.

The next experiment that most reduces architecture uncertainty is **oracle GT
crop recognition at true L0, L1, and L2 source resolution**. It directly tests
whether the identity information needed by any downstream recognizer exists at
each camera level. If resolution does not restore separability, an active-zoom
architecture is not justified; if it does, it establishes the need for zoom
and supplies the exact crops for the subsequent representation comparison.

## 1. Repository state and preservation

Initial state was exactly the expected frozen baseline:

- Branch: `drone-minimal-e2e-baseline`
- HEAD: `129564f855060ea373dc8421701f97631162a828`
- Tracking: `origin/drone-minimal-e2e-baseline`
- Worktree before the audit: clean
- Baseline checkpoint: 5,494,554 bytes, SHA-256
  `4c44e404e03673e8aa15fc85baf90f2d09b67d90c1fd94c9b6c9a4c1f35204ce`

The audit writes only to this new directory. It does not load the model, run
inference, train, deploy, call the hosted service, or modify prior evidence.
The 199 tracked files across the evaluator probes, scene audit, and three
motion experiment directories have aggregate path+content SHA-256
`aed49b1bac7a69700d63df247590801158aceb1aeee64469686409846b74eba9`.
Exact key-input hashes are in `preservation_manifest.json`.

## 2. Evidence inspected

The audit consumed:

- `training/MINIMAL_E2E_BASELINE.md`
- `training/artifacts/evaluation_results.json`, including all 235 predictions,
  per-frame timings, offline/realtime results, and per-class AP
- `training/artifacts/validation_results.json`
- `training/artifacts/training_metadata.json`, `training_args.yaml`,
  `training_results.csv`, `training_curves.png`, and the saved confusion plot
- all 25 `src/helsinki/annotations/frame_*.json` files and corresponding images
- `scene_audit/annotation_metrics.csv`, class/temporal summaries, galleries,
  and reviewed scene-audit report
- `detector.py`, `api.py`, `training/data_prep.py`,
  `training/evaluate_baseline.py`, DTOs, and evaluator implementation
- reviewed evaluator-probe, translation, similarity, and affine artifacts
- repository-wide search for the hosted attempt UUID, hosted logs, request
  records, validation traces, and server-side artifacts
- handoff-provided hosted facts: attempt
  `7b96ded345424b96b6f43ab28ee14bbf`, raw score
  `0.0016190390038119638`, working endpoint, and no evaluator errors

No historical hosted request/prediction log was found in the repository.

## 3. Matching method

All boxes are compared in original 3840×2160 coordinates. Within each frame,
the deterministic one-to-one association proceeds in this order:

1. correct class and IoU ≥ 0.50, ordered by confidence (evaluator-like TP);
2. wrong class and IoU ≥ 0.50 (localized identity error);
3. correct class and 0.10 ≤ IoU < 0.50 (localization error);
4. wrong class and 0.10 ≤ IoU < 0.50 (weak-overlap wrong class);
5. no remaining overlap: missed GT.

Unassigned predictions are separately marked as background FP (best GT IoU
<0.10), low-overlap FP (0.10–<0.50), or duplicate/alternative (≥0.50).
Every row also retains best-any, best-correct, and best-wrong overlaps so the
taxonomy can be reinterpreted without rerunning inference. The 0.10 threshold
is an audit association convention, not a proposed model threshold.

## 4. Prediction-level results

### GT outcomes (denominator 259)

| Outcome | Count | Rate |
|---|---:|---:|
| D — correct class and IoU ≥ 0.50 | 116 | 44.8% |
| A — missed entirely | 131 | 50.6% |
| B — localized at IoU ≥ 0.50, wrong class | 10 | 3.9% |
| C — correct class, IoU 0.10–<0.50 | 1 | 0.4% |
| G — wrong class, IoU 0.10–<0.50 | 1 | 0.4% |

Any-class localization recall at IoU ≥0.50 is 126/259 (48.6%). The 10-point
difference from correct detection is entirely identity confusion. Only one GT
has correct identity but inadequate localization, so localization is not the
dominant failure among objects that produce class evidence.

### Prediction outcomes (denominator 235)

| Outcome | Count |
|---|---:|
| Correct TP | 116 |
| Wrong-class IoU ≥0.50 | 10 |
| Correct-class low-IoU | 1 |
| Wrong-class low-IoU | 1 |
| Background FP | 48 |
| Low-overlap FP | 53 |
| Duplicate / alternative | 6 |

### Class results

| Class | GT | AP@.50 | Direct correct recall | Any-class localization recall |
|---|---:|---:|---:|---:|
| hangar | 6 | 0.663 | 66.7% | 66.7% |
| helicopter | 19 | 0.941 | 94.7% | 94.7% |
| jet_plane | 22 | 0.950 | 95.5% | 95.5% |
| large_launcher | 25 | 0.998 | 100.0% | 100.0% |
| large_tower | 19 | 0 | 0% | 0% |
| medium_launcher | 10 | 0 | 0% | 0% |
| medium_plane | 5 | 0 | 0% | 0% |
| mine_roller | 2 | 0 | 0% | 0% |
| small_launcher | 25 | 0 | 0% | 0% |
| small_plane | 9 | 0 | 0% | 0% |
| small_tower | 20 | 0.913 | 95.0% | 95.0% |
| ta-ta | 25 | 0 | 0% | 0% |
| tank | 25 | 0 | 0% | 32.0% |
| condor | 11 | 0.812 | 81.8% | 81.8% |
| jammer | 13 | 0 | 0% | 0% |
| spacecraft | 23 | 0.861 | 87.0% | 95.7% |

Nine classes are never correctly recognized. More strongly, **none of the 235
saved predictions uses any of those nine class IDs** at the deployed 0.01
confidence floor. This is absence of class evidence in the emitted outputs,
not an AP artifact.

The systematic confusion is narrow and interpretable:

- tank is localized in 8/25 frames but labeled spacecraft in 7 and small_tower
  in 1 (median assigned IoU 0.860; low confidences 0.018–0.064);
- spacecraft is labeled small_tower in frames 3 and 4 at high confidence
  0.787/0.791 and IoU 0.904/0.819;
- jammer has one weak overlap labeled jet_plane (IoU 0.132).

## 5. Size and resolution

| L0 short side | GT | Any-class IoU≥.50 | Correct class+IoU≥.50 |
|---|---:|---:|---:|
| <4 px | 10 | 0% | 0% |
| 4–<8 px | 50 | 0% | 0% |
| 8–<16 px | 121 | 40.5% | 32.2% |
| 16–<32 px | 66 | 98.5% | 98.5% |
| ≥32 px | 12 | 100% | 100% |

Target short side and correct detection have Pearson `r=0.694`. The transition
is stark: all 60 sub-8-pixel appearances are missed, while 77/78 appearances
at ≥16 pixels are correct. This confirms local L0 information starvation.

Size is not a sufficient explanation. The 8–<16 bin mixes 39 successes, 10
well-localized identity errors, and 72 other failures. `large_tower` is never
detected over 19 appearances even as its L0 short side reaches 15.75 px;
`tank` reaches 13 px and is sometimes localized but never named correctly;
meanwhile `small_tower` and `spacecraft` often succeed in the same size band.
The physical class names “large” and “small” are not image-size bins.

Boundary GT recall is 6/23 (26.1%) versus 110/236 (46.6%) off-boundary.
Boundary truncation is supported as a secondary risk, but this observational
comparison is confounded by class, size, and sequence entry/exit.

## 6. Confidence and ranking

True-positive confidence has median 0.879 and mean 0.761. Incorrect prediction
confidence has median 0.0178 and mean 0.0493. Confidence and best-GT IoU have
Pearson `r=0.742`, so there is useful local ranking signal.

It is not cleanly threshold-separable:

| Confidence | TP | Incorrect |
|---|---:|---:|
| 0.01–<0.02 | 3 | 64 |
| 0.02–<0.05 | 7 | 33 |
| 0.05–<0.10 | 3 | 10 |
| 0.10–<0.25 | 0 | 8 |
| 0.25–<0.50 | 4 | 2 |
| ≥0.50 | 99 | 2 |

The low-confidence true positives are real, well-localized hangar/condor cases;
the two high-confidence errors are spacecraft→small_tower. Raising the
threshold would remove many false positives but also erase useful rare-class
evidence. Lowering it might reveal additional raw candidates, but the saved
evidence contains no prediction whatsoever for the nine zero-AP class IDs.
Threshold tuning therefore cannot plausibly repair the dominant class-evidence
or hosted-transfer failure and is not the next action.

## 7. Temporal perception and memorization

Seven classes are recognized at least once. For the strong classes, perception
is stable on the repeated Helsinki instance: large_launcher 25/25,
small_tower 19/20, helicopter 18/19, jet_plane 21/22, spacecraft 20/23,
condor 9/11, and hangar 4/6. Their longest correct-detection gaps are 0–2
frames. Temporal fusion could bridge those brief gaps.

It cannot create identity for the other nine classes: their correct-detection
gap equals their entire visible lifetime (up to 25 frames). Tank demonstrates
that temporal localization without correct identity can instead reinforce a
systematic wrong label. Temporal state must therefore be downstream of a
recognizer that first establishes trustworthy identity.

Confidence often covaries with size, frame, and position for successful
objects (for example helicopter confidence versus short side `r=0.693`), but
these variables co-evolve on a single trajectory. The dataset cannot separate
object texture, physical-instance texture, position, trajectory, and
surrounding background. The hosted gap plus one-instance-per-class design
supports reference-instance/scene memorization as the central diagnosis, but
background memorization specifically remains plausible rather than confirmed.
No synthetic context probe was promoted to a generalization test in this audit.

## 8. Training behavior

Training and “validation” contain the same 25 frames. The curves show
continued same-sequence fitting, not generalization:

- best and final same-sequence mAP@.50: 0.43021 at epoch 60;
- mAP@.50 rises from 0.36218 at epoch 50 to 0.43021 at epoch 60;
- mean train box loss falls from 3.459 (epochs 1–10) to 1.621 (51–60);
- mean train class loss falls from 7.043 to 3.082;
- mean same-sequence val box/class losses fall from 2.978/6.003 to
  1.580/2.957;
- final Ultralytics diagnostic precision/recall is 0.963/0.294.

There is no observed late divergence on the repeated frames and no evidence of
confidence collapse. That does not contradict overfit: the experiment lacks an
independent validation distribution. Mosaic was configured at 1.0 and closed
for the last 10 epochs; standard HSV, translation, scale, flip, and erasing
augmentations were enabled. Class count correlates only weakly with AP
(`r=0.279`): 25-example classes include both perfect (`large_launcher`) and
zero-recall (`small_launcher`, `ta-ta`, `tank`) outcomes. Imbalance may hurt
rare classes but is not supported as the primary failure.

## 9. Hosted evidence and limits

The hosted endpoint connected, returned schema-valid predictions, completed
validation, and recorded no errors. Together with exact checkpoint identity,
this rejects gross packaging, API, and schema failure as the main explanation.

**Hosted GT-level failure attribution is not observable from the available
evidence.** We cannot determine hosted per-class AP/recall, missed targets,
localization quality, class confusion, target sizes, frame count, camera views,
prediction counts, confidence distribution, or whether the portal's single
`small_tower` sample overlaps a GT. The raw hosted score is valid evidence of
catastrophic transfer failure, but it cannot identify its internal composition.

## 10. Quantitative failure taxonomy

| Failure mode | Evidence | Severity | Diagnosis | Likely remedy / gate |
|---|---|---|---|---|
| L0 information loss | 60/60 sub-8 px missed; 65/66 correct at 16–<32 | Critical | **CONFIRMED locally; SUPPORTED hosted contribution** | Oracle crop × resolution |
| Physical-instance/reference memorization | one instance/class; 0.3837 local vs 0.001619 hosted | Critical | **SUPPORTED** | Transfer-oriented crop comparison |
| Background/context memorization | compatible with transfer gap; no intervention | Potentially critical | **PLAUSIBLE** | Controlled context probe |
| Class confusion | 10 localized wrong: 8 tank, 2 spacecraft | High for affected classes | **CONFIRMED** | Transfer recognition/prototypes |
| Localization weakness | only 1 correct-class low-IoU GT | Low as primary cause | **NOT SUPPORTED AS PRIMARY** | L0 objectness test, not box tuning |
| Confidence/ranking weakness | 13 TP and 107 incorrect below .10; 2 errors ≥.50 | Moderate/secondary | **CONFIRMED** | Quality gates only after perception |
| Class imbalance | count/AP `r=0.279`; frequent zero classes | Secondary/unknown | **NOT SUPPORTED AS PRIMARY** | No dedicated experiment yet |
| Boundary truncation | 26.1% vs 46.6% recall, confounded | Moderate | **SUPPORTED** | Stratify future tests; later state |
| Temporal intermittency | 0–2 gaps for recognized classes; 9 never recognized | Moderate | **CONFIRMED** | State only after correct seeds |
| Runtime skipping | local realtime sends 8/25; exact sent predictions | High for local realtime | **CONFIRMED** | Later latency/scheduler work |
| API/infrastructure | hosted worked and completed without errors | Low | **NOT SUPPORTED** | None indicated |
| Hosted GT attribution | no hosted GT/logs | Limitation | **NOT OBSERVABLE** | Local discriminating experiments |

The full machine-readable table is `failure_taxonomy.csv`.

## 11. Highest-information next experiments

1. **Oracle GT crop × true L0/L1/L2 recognition (first).** Use identical GT
   geometry and deterministic padding at each true source resolution. Measure
   per-class separability. This decides whether zoom is necessary/sufficient.
2. **Identical-crop representation comparison.** On exactly the same crops,
   compare current supervised YOLO-derived features, frozen pretrained visual
   embeddings, exemplar/prototype matching, and a simple hybrid only if the
   individual signals justify it. This decides how identity should transfer.
3. **L0 localization-only/objectness.** Ignore class identity and measure
   class-agnostic proposal recall versus size at a fixed proposal budget. This
   decides whether L0 can reliably point the camera at targets.
4. **Controlled context-dependence probe.** Apply small deterministic object
   translations, object masking, and matched context controls. This separates
   object evidence from Helsinki background/position evidence; it remains a
   forensic probe, not a benchmark.

These are deliberately ordered. The first experiment is the single largest
uncertainty reducer; the second determines the recognizer; the third determines
whether active zoom is discoverable; the fourth resolves the remaining
memorization mechanism.

## 12. Architecture V2 hypothesis — not implemented

| Proposed component | Measured failure addressed | Evidence | Must pass before inclusion | Simplest fallback |
|---|---|---|---|---|
| L0 discovery/objectness | 131 misses; need zoom proposals | Identity and localization can be separated | useful class-agnostic recall at fixed proposal budget | sparse deterministic scan/refresh |
| Active L1/L2 zoom and scheduler | L0 information starvation | 60/60 sub-8 px missed; prior audit shows true added source information | oracle resolution gives material recognition gain | bounded L1-only refresh; L2 only if marginal gain proves out |
| Frozen embedding/prototype recognizer; supervised hybrid only if earned | hosted collapse, nine absent IDs, identity confusion | reference fit does not transfer; only 7 IDs emitted | win identical-crop held-control comparison with useful ranking | best frozen representation + nearest prototype |
| Affine GMC + temporal state with staleness/uncertainty | brief gaps, boundaries, skipped observations | reviewed affine one-step 97.94% IoU≥.50; 7 classes have valid intermittent seeds | perception supplies correct seeds and temporal simulation improves recall without identity contamination | short-TTL affine hold-last only |
| Confidence quality gates | mixed low-confidence tail and two high-confidence confusions | TP median .879 vs incorrect .0178, but tails overlap | held-control calibration improves risk/coverage | preserve raw score plus conservative uncertainty flag |

The causal order is perception → proposal/zoom → identity → temporal state.
Affine motion remains the correct GMC target, but it cannot manufacture an
identity that the recognizer never emits.

## 13. Research decision

**No Research V2 needed yet; next empirical experiments are already defined.**
The audit did not discover a new problem that requires broad literature work.
It measured the anticipated tiny-object and one-instance transfer risks much
more sharply and identified the exact experiments needed to choose V2.

## 14. Reproducibility

`run_audit.py` is deterministic and uses only frozen inputs. The audit ran with
Python 3.12.14, NumPy 2.3.5, and Pillow 12.3.0 on Windows 11. It was rerun from
the same inputs; output tables, JSON, and figures are regenerated in place.
The script records source commit, package versions, key input hashes, protected
artifact aggregate hash, matching thresholds, and all summary statistics.

The visual set answers the requested questions:

- `01_gt_count_by_class.png` — class support
- `02_ap_recall_by_class.png` — AP versus direct recall
- `03_recall_by_size.png` — L0 size dependence
- `04_confidence_correct_incorrect.png` — ranking separation
- `05_confidence_vs_size.png` — confidence/size relationship
- `06_failure_type_counts.png` — GT and prediction error taxonomy
- `07_confusion_no_detection.png` — confusion with no-detection separated
- `08_temporal_timeline.png` — per-object temporal consistency
- `09_representative_overlays.png` — success, tiny miss, large_tower miss,
  wrong class, localization failure, and background false positive

All underlying rows are retained in the CSVs; figures are not the evidence of
record.

