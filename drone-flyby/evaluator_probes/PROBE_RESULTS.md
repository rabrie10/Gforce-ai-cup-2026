# Evaluator probe results

Measured against the supplied implementation. No detector, tracker, training, camera-policy design, or production-solution changes are included.

# Environment

| Item | Recorded value |
| --- | --- |
| Run UTC | 2026-09-17T21:36:45.059667+00:00 |
| Commit | 2e6979e1980cff9fb9a33bb3e4869217700d1009 |
| Branch | drone-evaluator-probes |
| Python | 3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)] |
| Platform | Windows-11-10.0.26200-SP0 |
| Evaluator monotonic clock | {'implementation': 'GetTickCount64()', 'monotonic': True, 'adjustable': False, 'resolution': 0.015625} |
| Source/data unchanged during suite | True |
| faster-coco-eval | 1.8.0 |
| numpy | 2.5.3 |
| opencv-python | 5.0.0.93 |
| pydantic | 2.13.5 |
| requests | 2.34.2 |
| fastapi | 0.141.1 |
| uvicorn | 0.53.0 |

Git state at measurement start:

```text
?? drone-flyby/evaluator_probes/
```

The repository was clean before this task. Created `drone-evaluator-probes` from `codex/drone-evaluator-probes` at the commit above. All task files are under `drone-flyby/evaluator_probes/`; authoritative files are unchanged. The JSON stores SHA-256 fingerprints of the five executable evaluator/protocol/helper/endpoint files and all 51 Helsinki data files. The installed scorer is **1.8.0**, permitted by the supplied `>=1.7.2,<2` requirement; the README comments mentioning 1.7.2 are not an installed-version check.

Environment recovery: `python` was unavailable, `py -3.13` referenced a missing `C:\Python313`, and the `uv` launcher was broken. Used bundled Python 3.12.14 to create the ignored `.venv`. Initial sandboxed pip networking failed; installation succeeded with network permission. No evaluator changes were needed. Exact bootstrap command:

```powershell
& "C:\Users\mbnas\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv drone-flyby/.venv
& drone-flyby/.venv/Scripts/python.exe -m pip install -r drone-flyby/requirements.txt
```

# Method

Pure metric probes call the unchanged `local_evaluator.score`. Synthetic scenes are real temporary PNG/annotation directories read through the official loaders. They use a 200x200 `hangar` box at `[100,100,300,300]`; controlled FPs use `[600,600,800,800]`. Protocol probes call the unchanged `replay` against a deterministic local HTTP server and use official global-coordinate helpers and DTO validation. A-C and G have one GT per frame; D also includes two same-class GTs; H/I have two good detections of different classes in each of two frames, so a malformed addition can demonstrably discard good detections.

Timing sweeps use all 25 Helsinki images, a serial evaluator, real HTTP, actual endpoint sleeps, and the real clock. No simulated-latency flag or fake clock is used. A read-only Python trace records elapsed time and selected indices immediately after the evaluator calculation. A no-trace control and separately timed image preparation check its interpretation. The synthetic timing follow-up uses the same full-resolution rendering path with simple PNG content; it does not stand in for the Helsinki performance measurement.

Each deterministic case changes its named factor. The monotonic confidence transform is squaring. Missing-frame recall is independently counted from retained GT annotations, and IoU uses independent intersection/union arithmetic. COCO precision samples are also read from the actual scorer in C. Timing runs are sequential single runs per condition, not statistical estimates. On the recorded Python 3.12 Windows runtime, the evaluator clock is `GetTickCount64()` with 15.625 ms resolution; sub-tick distinctions in sleep targets such as 330 versus 333 ms cannot establish precise boundaries. The clock was not replaced. Differences between nearby delays must not be treated as a universal performance boundary. Raw counters, RTTs, per-class scores, frame indices, camera states, feedback, diagnostic text, and recurrence observations are in `probe_results.json`.

# Results Summary

| Probe | Hypothesis | Observed result | Confirmed? | Strategic importance |
| --- | --- | --- | --- | --- |
| Oracle/H1 | Perfect GT gives 1 | 1.000000 | Yes | Environment gate |
| A/H2 | Off-crop detections count equally | In/out crop both 1.000000 | True | Global reporting |
| B/H3 | Inclusive IoU .50 | 0.000000, 1.000000, 1.000000, 1.000000, 1.000000 | True | Matching boundary |
| C/H4 | FP rank changes AP | TP first 1; FP first .5; square unchanged | True | Confidence order |
| D/H5 | Duplicate penalty depends on ranking | Single GT: all 1; duplicate before second TP: 0.834983 | Refined | No fixed duplicate penalty |
| E/H6 | Missing frames cost recall | Every second: 0.478960 | Yes | Coverage |
| F/H7 | Latency skips frames | Measured jumps; preprocessing dominates on this host | Yes, with clock refinement | End-to-end throughput |
| G/H8 | 100 detections retained per class/image | Rank 99 / 100 / 101: 0.010101 / 0.010000 / 0.000000 | Yes | Rank cap |
| H/H9 | Bad responses lose all detections | Invalid first responses lose both good boxes | Yes; ID mismatches are schema-valid | Whole-response integrity |
| I/H9 | Illegal camera geometry preserves detections | 1.000000 | Yes | Separate failure effects |
| J/H10 | Feedback persists until valid command | Present frames 1-3; absent frame 4 | Yes | Feedback lifecycle |
| K/H7 | Late acceptance differs from timeout | 3000 ms accepted; 3500 ms times out | Yes | Throughput versus deadline |
| L | Absent allowed classes excluded | Oracle plus absent-class FP: 1.000000 | Yes, synthetic scope | Evaluated category set |

# Detailed Results

## Oracle — environment gate

**Question/hypothesis:** Does supplied GT score 1.000?

**Setup:** Unmodified full Helsinki scene, supplied oracle CLI.

**Exact command:**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/local_evaluator.py --oracle
```

**Expected:** 1.000 overall and all classes. **Observed output:**

```text
Scoring the ground truth for helsinki ...

AP@0.50 by class
  hangar           1.000
  helicopter       1.000
  jet_plane        1.000
  large_launcher   1.000
  large_tower      1.000
  medium_launcher  1.000
  medium_plane     1.000
  mine_roller      1.000
  small_launcher   1.000
  small_plane      1.000
  small_tower      1.000
  ta-ta            1.000
  tank             1.000
  condor           1.000
  jammer           1.000
  spacecraft       1.000

COCO mAP@0.50: 1.000
```

**Matched:** Yes. **Interpretation/implication:** Local scorer and supplied annotations are consistent enough to proceed. **Uncertainty:** This is not proof of hosted scorer parity.

## A — Off-crop global predictions

**Question:** Does current crop restrict current scoring?

**Hypothesis:** H2: identical correct global boxes score equally inside and outside the crop.

**Setup:** Three-frame synthetic scene; frame 0 commands L1 center (960,540) or (2880,1620). Frames 1-2 contain the same global box; it intersects only the first crop. All predictions are identical.

**Exact command (complete suite, JSON key `A`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** Equal AP, ideally 1.0.

**Observed:**

| Case | AP | Crop intersects GT, frames 0/1/2 |
| --- | --- | --- |
| in_crop | 1.000000 | [True, True, True] |
| off_crop | 1.000000 | [True, False, False] |

**Matched expectation?** True

**Interpretation:** The actual replay accepts, converts, and scores the off-crop boxes with no crop restriction.

**Possible implication:** Responses can carry whole-frame information beyond the current observation.

**Remaining uncertainty:** This tests scoring eligibility, not the ability to estimate an unseen object.

## B — IoU boundary

**Question:** Where is the match threshold?

**Hypothesis:** H3: .499 fails; .500 and .501 match.

**Setup:** One GT box; prediction has the same height and left edge and width 200 times target IoU. At .500 all corners are integers. Other fixture geometry and confidence stay fixed.

**Exact command (complete suite, JSON key `B`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** AP 0,1,1,1,1 for IoUs .499,.500,.501,.60,.90.

**Observed:**

| Target IoU | Independently computed IoU | AP |
| --- | --- | --- |
| 0.499 | 0.4990000000000001 | 0.000000 |
| 0.5 | 0.5 | 1.000000 |
| 0.501 | 0.5009999999999999 | 1.000000 |
| 0.6 | 0.6 | 1.000000 |
| 0.9 | 0.9 | 1.000000 |

**Matched expectation?** True

**Interpretation:** The inclusive .50 threshold is measured. Isolated .60 and .90 matches have equal AP.

**Possible implication:** Extra localization beyond the matching threshold gives no direct gain in this controlled case.

**Remaining uncertainty:** Multi-object association ambiguities and floating-point perturbations around .50 are not exhausted.

## C — Confidence ranking

**Question:** Can the same FP be harmless or costly depending on confidence?

**Hypothesis:** H4: TP-first may avoid AP loss; FP-first lowers AP; monotonic confidence transforms preserve it.

**Setup:** One GT, one perfect TP, one disjoint FP. Only scores change between .9/.1 and .1/.9, then each pair is squared. Read actual COCO precision tensor on return from scorer.

**Exact command (complete suite, JSON key `C`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** TP-first 1; FP-first .5; squared variants unchanged.

**Observed:**

| Case | AP |
| --- | --- |
| TP_first | 1.000000 |
| FP_first | 0.500000 |
| TP_first_squared | 1.000000 |
| FP_first_squared | 0.500000 |

| Actual tensor | Recall samples | Min precision | Max precision | Maximum recall |
| --- | --- | --- | --- | --- |
| TP_first | 101 | 1.0 | 1.0 | 1.0 |
| FP_first | 101 | 0.5 | 0.5 | 1.0 |

**Matched expectation?** True

**Interpretation:** COCO evaluates a confidence-sorted prefix, envelopes precision from the right, and samples 101 recall points. TP-first reaches recall 1 with precision 1 before the FP; every sampled precision is 1. FP-first reaches recall 1 at precision .5; every sampled precision is .5. The measured precision arrays support the AP difference directly.

**Possible implication:** Confidence ranking can affect AP substantially; an extra FP has no universal fixed AP cost.

**Remaining uncertainty:** A late FP need not be harmless when useful detections remain later in the class ranking or when maxDets truncates them. Ties are not tested.

## D — Duplicate ranking

**Question:** When do duplicate boxes reduce AP?

**Hypothesis:** H5: later matches to already-matched GT are FPs, with rank-dependent cost.

**Setup:** Single-object cases reverse scores on two identical perfect boxes; a two-object same-class case moves only duplicate confidence from .1 to .8 while useful TP scores stay .9/.5.

**Exact command (complete suite, JSON key `D`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** Single useful TP followed by duplicates may retain AP 1; duplicates ahead of remaining useful TPs can cost AP.

**Observed:**

| Case | AP |
| --- | --- |
| one | 1.000000 |
| duplicate_below | 1.000000 |
| duplicate_above | 1.000000 |
| five_duplicates_below | 1.000000 |
| duplicate_before_second_TP | 0.834983 |
| duplicate_after_all_TPs | 1.000000 |

**Matched expectation?** Confirmed with a material refinement of the phrase “duplicate above useful detection”.

**Interpretation:** There is no preassigned useful copy: the first matching duplicate becomes the TP. With two GTs, TP/duplicate/TP gives precision 1 through recall .50 and 2/3 thereafter. The 101-point average is (51*1 + 50*2/3)/101 = .83498349835, matching measurement.

**Possible implication:** Evaluate duplicate placement relative to remaining useful detections, not duplicate count alone.

**Remaining uncertainty:** Overlapping multiple GTs and large duplicate sets that reach maxDets are outside this fixture.

## E — Missing frames

**Question:** Do omitted predictions remove GT from evaluation?

**Hypothesis:** H6: GT stays, so omitted frames reduce attainable recall and AP.

**Setup:** Start with full Helsinki oracle; delete only the frame keys shown. All retained detections remain perfect.

**Exact command (complete suite, JSON key `E`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** Less recall and AP whenever removed frames contain GT; class effects depend on visibility.

**Observed:**

| Removed pattern | Removed frames | AP | Retained / total GT | Micro recall |
| --- | --- | --- | --- | --- |
| none | [] | 1.000000 | 259/259 | 1.000000 |
| frame_0 | [0] | 0.930074 | 248/259 | 0.957529 |
| every_second_even | [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24] | 0.478960 | 124/259 | 0.478764 |
| every_third_zero | [0, 3, 6, 9, 12, 15, 18, 21, 24] | 0.634901 | 166/259 | 0.640927 |
| last_five | [20, 21, 22, 23, 24] | 0.771658 | 212/259 | 0.818533 |

Per-class **AP / independently counted recall**:

| Class | none | frame_0 | every_second_even | every_third_zero | last_five |
| --- | --- | --- | --- | --- | --- |
| hangar | 1.000000 / 1.000000 | 1.000000 / 1.000000 | 0.504950 / 0.500000 | 0.663366 / 0.666667 | 0.168317 / 0.166667 |
| helicopter | 1.000000 / 1.000000 | 0.940594 / 0.947368 | 0.475248 / 0.473684 | 0.633663 / 0.631579 | 1.000000 / 1.000000 |
| jet_plane | 1.000000 / 1.000000 | 1.000000 / 1.000000 | 0.504950 / 0.500000 | 0.633663 / 0.636364 | 0.772277 / 0.772727 |
| large_launcher | 1.000000 / 1.000000 | 0.960396 / 0.960000 | 0.485149 / 0.480000 | 0.643564 / 0.640000 | 0.801980 / 0.800000 |
| large_tower | 1.000000 / 1.000000 | 1.000000 / 1.000000 | 0.475248 / 0.473684 | 0.633663 / 0.631579 | 0.732673 / 0.736842 |
| medium_launcher | 1.000000 / 1.000000 | 1.000000 / 1.000000 | 0.504950 / 0.500000 | 0.693069 / 0.700000 | 0.603960 / 0.600000 |
| medium_plane | 1.000000 / 1.000000 | 1.000000 / 1.000000 | 0.405941 / 0.400000 | 0.603960 / 0.600000 | 0.000000 / 0.000000 |
| mine_roller | 1.000000 / 1.000000 | 0.504950 / 0.500000 | 0.504950 / 0.500000 | 0.504950 / 0.500000 | 1.000000 / 1.000000 |
| small_launcher | 1.000000 / 1.000000 | 0.960396 / 0.960000 | 0.485149 / 0.480000 | 0.643564 / 0.640000 | 0.801980 / 0.800000 |
| small_plane | 1.000000 / 1.000000 | 0.881188 / 0.888889 | 0.445545 / 0.444444 | 0.663366 / 0.666667 | 1.000000 / 1.000000 |
| small_tower | 1.000000 / 1.000000 | 0.940594 / 0.950000 | 0.504950 / 0.500000 | 0.653465 / 0.650000 | 1.000000 / 1.000000 |
| ta-ta | 1.000000 / 1.000000 | 0.960396 / 0.960000 | 0.485149 / 0.480000 | 0.643564 / 0.640000 | 0.801980 / 0.800000 |
| tank | 1.000000 / 1.000000 | 0.960396 / 0.960000 | 0.485149 / 0.480000 | 0.643564 / 0.640000 | 0.801980 / 0.800000 |
| condor | 1.000000 / 1.000000 | 0.900990 / 0.909091 | 0.455446 / 0.454545 | 0.633663 / 0.636364 | 1.000000 / 1.000000 |
| jammer | 1.000000 / 1.000000 | 0.920792 / 0.923077 | 0.465347 / 0.461538 | 0.613861 / 0.615385 | 1.000000 / 1.000000 |
| spacecraft | 1.000000 / 1.000000 | 0.950495 / 0.956522 | 0.475248 / 0.478261 | 0.653465 / 0.652174 | 0.861386 / 0.869565 |

**Matched expectation?** True

**Interpretation:** Omitted frames remain in the GT denominator. The reported metric macro-averages class AP; it is not pooled recall. COCO recall-grid interpolation also means AP does not always equal continuous recall, even with only exact retained detections.

**Possible implication:** Missed observation opportunities have class- and frame-dependent metric cost.

**Remaining uncertainty:** These loss patterns and class frequencies describe Helsinki, not an unknown hosted scene.

## F — Realtime latency sweep

**Question:** How does response latency change the sent sequence?

**Hypothesis:** H7: serial processing loses intermediate frames; request deadline differs from throughput budget.

**Setup:** Actual endpoint sleep at every listed delay; all 25 Helsinki frames; full image I/O, crop, PNG, Base64, HTTP, DTO and camera logic. Follow-ups measure preparation without tracing and repeat zero-delay Helsinki without tracing because the first sweep already skipped frames at zero endpoint sleep. Simple-image timing isolates content-dependent overhead.

**Exact command (complete suite, JSON key `F`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** Longer total processing time generally causes skips; exact counts must follow the implemented recurrence.

**Observed:**

| Sleep ms | Total | Sent | Skipped | Unanswered | Accepted | Timeouts | RTT mean / median / max ms | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 25 | 12 | 13 | 0 | 12 | 0 | 33.9 / 31.0 / 47.0 | 0.478960 |
| 250 | 25 | 9 | 16 | 0 | 9 | 0 | 283.0 / 281.0 / 297.0 | 0.368812 |
| 300 | 25 | 8 | 17 | 0 | 8 | 0 | 332.0 / 328.0 / 344.0 | 0.329827 |
| 320 | 25 | 9 | 16 | 0 | 9 | 0 | 352.3 / 344.0 / 375.0 | 0.368812 |
| 330 | 25 | 9 | 16 | 0 | 9 | 0 | 352.3 / 359.0 / 360.0 | 0.368812 |
| 333 | 25 | 9 | 16 | 0 | 9 | 0 | 363.0 / 360.0 / 375.0 | 0.368812 |
| 340 | 25 | 8 | 17 | 0 | 8 | 0 | 371.0 / 375.0 / 375.0 | 0.337252 |
| 350 | 25 | 8 | 17 | 0 | 8 | 0 | 381.0 / 375.0 / 391.0 | 0.332302 |
| 400 | 25 | 8 | 17 | 0 | 8 | 0 | 423.8 / 422.0 / 438.0 | 0.337252 |
| 500 | 25 | 7 | 18 | 0 | 7 | 0 | 527.0 / 531.0 / 547.0 | 0.301361 |
| 650 | 25 | 6 | 19 | 0 | 6 | 0 | 677.2 / 672.0 / 688.0 | 0.264851 |
| 700 | 25 | 6 | 19 | 0 | 6 | 0 | 723.8 / 719.0 / 735.0 | 0.264851 |
| 1000 | 25 | 5 | 20 | 0 | 5 | 0 | 1028.2 / 1031.0 / 1032.0 | 0.228342 |

Independent Helsinki image preparation (25 frames, no trace):

| Stage | Mean ms | Median ms | Max ms |
| --- | --- | --- | --- |
| load_ms | 476.2 | 469.0 | 578.0 |
| render_ms | 201.3 | 203.0 | 235.0 |
| total_ms | 677.5 | 656.0 | 812.0 |

Helsinki no-trace zero-delay control:

| Sleep ms | Total | Sent | Skipped | Unanswered | Accepted | Timeouts | RTT mean / median / max ms | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 25 | 13 | 12 | 0 | 13 | 0 | 25.2 / 31.0 / 46.0 | 0.527228 |

Same replay path, simple synthetic PNGs:

| Sleep ms | Total | Sent | Skipped | Unanswered | Accepted | Timeouts | RTT mean / median / max ms | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 25 | 25 | 0 | 0 | 25 | 0 | 8.8 / 15.0 / 16.0 | 1.000000 |
| 200 | 25 | 21 | 4 | 0 | 21 | 0 | 209.1 / 204.0 / 219.0 | 0.841584 |
| 250 | 25 | 19 | 6 | 0 | 19 | 0 | 256.5 / 250.0 / 266.0 | 0.762376 |
| 300 | 25 | 17 | 8 | 0 | 17 | 0 | 305.1 / 312.0 / 313.0 | 0.683168 |
| 333 | 25 | 16 | 9 | 0 | 16 | 0 | 340.0 / 344.0 / 344.0 | 0.643564 |
| 350 | 25 | 16 | 9 | 0 | 16 | 0 | 358.2 / 359.0 / 375.0 | 0.643564 |

Synthetic requests sent **before** their nominal index/3 emission time:

| Sleep ms | Early requests | Replay wall seconds |
| --- | --- | --- |
| 0 | 23 | 4.672 |
| 200 | 0 | 8.469 |
| 250 | 0 | 8.531 |
| 300 | 0 | 8.719 |
| 333 | 0 | 8.485 |
| 350 | 0 | 8.563 |

All observed recurrence and frame-accounting comparisons agree: **True**.

**Matched expectation?** Latency skipping confirmed; fixed 333 ms endpoint-only threshold and literal emission pacing require correction.

**Interpretation:** `next = max(current+1, floor(elapsed/(1/3)))`; skipped increments by `max(0, min(next,N) - (current+1))`. Exact elapsed/next-index pairs are recorded for each iteration and independently recomputed. Elapsed includes preparation before POST, while RTT excludes it. The baseline and independent preparation measurements show why even a fast endpoint cannot guarantee no skips on this host. The synthetic early-request measurements expose a second detail: this local loop has no sleep to pace fast requests to the next nominal emission. `current+1` can send future frames early. Endpoint sleeps near 333 ms therefore do not define a universal cliff. Slight score increases at larger delay can reflect jitter and which class-bearing frames survive.

**Possible implication:** Measure end-to-end throughput and actual frame indices; separate it from the request deadline.

**Remaining uncertainty:** Single-run timing on this Windows host and a 15.625 ms monotonic clock; no hosted network, production capture scheduling, or hosted pacing parity is established. Tracing overhead is included except the marked no-trace control.

## G — maxDets boundary

**Question:** Is a rank-101 valid detection retained?

**Hypothesis:** H8: only the first 100 detections per image/class enter the reported AP configuration.

**Setup:** 101 same-class detections with strictly descending confidence. Exactly one box is the perfect GT box; 100 disjoint FPs occupy the other ranks. Move the useful box to rank 99/100/101.

**Exact command (complete suite, JSON key `G`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** Rank 99 and 100 can match, with AP reduced by preceding FPs; rank 101 cannot.

**Observed:**

| Useful rank | Detections | AP |
| --- | --- | --- |
| 99 | 101 | 0.010101 |
| 100 | 101 | 0.010000 |
| 101 | 101 | 0.000000 |

Installed COCO default `maxDets`: `[1, 10, 100]`; the official score reads the final maxDets axis.

**Matched expectation?** True

**Interpretation:** Rank 99 gives 1/99, rank 100 gives 1/100, rank 101 gives zero. This is a scoring truncation, distinct from the 500-annotation response-validation limit.

**Possible implication:** Useful detections need sufficient within-image/class rank to survive the cap.

**Remaining uncertainty:** Measured for the installed faster-coco-eval version and this evaluator configuration.

## H — Malformed responses

**Question:** Which errors discard the entire response?

**Hypothesis:** H9: schema-invalid content loses all detections; mismatched IDs also invalidate the response.

**Setup:** Mutate frame 0 only in a two-frame scene with hangar and tank GT in both. Keep both valid annotations, add one malformed annotation where applicable, and include an otherwise legal L1 move. Frame 1 is always a clean oracle response with no command.

**Exact command (complete suite, JSON key `H`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** Invalid responses lose both good detections at frame 0, refuse processing of the camera command, and increment invalid/unanswered counters. Clean frame 1 still scores.

**Observed:**

| Case | Schema valid | Frame 0 scored | Accepted / 2 | Unanswered | Invalid responses | Moves applied | Moves rejected | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid_control | True | True | 2 | 0 | 0 | 1 | 0 | 1.000000 |
| bbox_reversed | False | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| bbox_zero_width | False | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| bbox_out_of_range | False | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| nan_string | False | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| infinity_string | False | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| infinity_numeric_overflow | False | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| invalid_class | False | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| unexpected_key | False | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| wrong_request_id | True | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| wrong_frame | True | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| 501_annotations | False | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| float_camera | False | False | 1 | 1 | 1 | 0 | 0 | 0.504950 |
| illegal_geometry | True | True | 2 | 0 | 0 | 0 | 1 | 1.000000 |

Both frames are sent and none skipped in every case; timeouts and HTTP errors are zero. For invalid responses, only frame 1 retains its two detections. Full validation messages and next-request camera states are in JSON. Numeric NaN and Infinity fail direct DTO validation with “bbox coordinates must be finite”; a strict JSON encoder refuses them. Legal JSON strings `"NaN"`/`"Infinity"` fail numeric schema types. The legal JSON numeric exponent `1e999` parses to infinity here and is rejected by the finite-coordinate validator.

**Matched expectation?** Confirmed. Wrong request_id and frame are schema-valid but fail contextual replay checks.

**Interpretation:** Schema validation and request identity validation are separate stages, both capable of discarding the full response. Invalid camera field types invalidate detections too. No nonstandard numeric NaN JSON was emitted, so no HTTP claim is made for that encoding.

**Possible implication:** Response integrity includes every annotation, field type, unknown key and echoed request identity.

**Remaining uncertainty:** Other servers or JSON parsers may reject overflow earlier; numeric NaN has no standard JSON representation.

## I — Schema-valid illegal camera geometry

**Question:** Can a rejected move preserve current detections?

**Hypothesis:** H9: geometric camera rejection is separate from response rejection.

**Setup:** Compare L1 center_x=960 (legal), center_x=0 (integer but out of bounds), and center_x=960.0 (float). All carry identical two-object predictions.

**Exact command (complete suite, JSON key `I`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** Illegal geometry preserves AP and old L0 view; float invalidates the first response.

**Observed:**

| Case | AP | Next request level | Next center | Next feedback present |
| --- | --- | --- | --- | --- |
| valid_control | 1.000000 | 1 | (960, 540) | False |
| illegal_geometry | 1.000000 | 0 | (1920, 1080) | True |
| float_camera | 0.504950 | 0 | (1920, 1080) | False |

**Matched expectation?** Confirmed.

**Interpretation:** Integer geometry error retains the frame-0 detections and reports one invalid command; the float is a response-schema error, so its camera command is never attempted.

**Possible implication:** Camera geometry and wire-schema correctness have different consequences for score.

**Remaining uncertainty:** This controlled comparison tests an out-of-bounds center, not every geometric rejection branch.

## J — Feedback persistence

**Question:** When does feedback appear and clear?

**Hypothesis:** H10: rejection feedback repeats until a valid command is accepted.

**Setup:** Frame 0 requests illegal L1 center (0,540). Frames 1-2 omit the requested view. Frame 3 requests legal L1 (960,540). Frame 4 omits it again.

**Exact command (complete suite, JSON key `J`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** No feedback at frame 0; frame-0 feedback in requests 1,2,3; cleared in request 4.

**Observed:**

| Request frame | Current level | Feedback source frame | Response requested view |
| --- | --- | --- | --- |
| 0 | 0 | None | {'resolution_level': 1, 'center_x': 0, 'center_y': 540} |
| 1 | 0 | 0 | None |
| 2 | 0 | 0 | None |
| 3 | 0 | 0 | {'resolution_level': 1, 'center_x': 960, 'center_y': 540} |
| 4 | 1 | None | None |

AP 1.000000; accepted 5; applied commands 1; rejected commands 1.

**Matched expectation?** Confirmed.

**Interpretation:** Feedback belongs to the request built before processing that frame response. The request carrying the successful repair still contains the old feedback; the next request clears it.

**Possible implication:** Treat feedback as persistent state, and observe repair acknowledgment on the following request.

**Remaining uncertainty:** This frame-by-frame measurement is offline; persistence across actual skipped frames is not separately measured.

## K — Late response versus timeout

**Question:** Is a response below the deadline accepted despite skipping?

**Hypothesis:** H7: late valid responses count for their original frame; timeout loses them.

**Setup:** Full Helsinki realtime replay, endpoint sleeps 3000 or 3500 ms; configured requests timeout is 10/3 seconds.

**Exact command (complete suite, JSON key `K`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** 3000 ms valid answers accepted with skips; 3500 ms requests time out and become unanswered.

**Observed:**

| Sleep ms | Total | Sent | Skipped | Unanswered | Accepted | Timeouts | RTT mean / median / max ms | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3000 | 25 | 3 | 22 | 0 | 3 | 0 | 3031.3 / 3031.0 / 3032.0 | 0.152847 |
| 3500 | 25 | 3 | 22 | 3 | 0 | 3 | 3354.0 / 3359.0 / 3359.0 | 0.000000 |

```text
frame 0: no answer within 3333 ms
frame 12: no answer within 3333 ms
frame 24: no answer within 3333 ms
```

**Matched expectation?** Confirmed.

**Interpretation:** The timeout applies to the HTTP operation; total frame-loop duration can exceed 3333 ms because image preparation precedes POST. A late response under that HTTP timeout still contributes predictions to its original frame; a timeout contributes none.

**Possible implication:** Separate deadline reliability from the rate needed to preserve frame opportunities.

**Remaining uncertainty:** This is a non-streaming loopback server. Requests timeout/network semantics and remote scheduling do not establish a strict hosted end-to-end wall-clock deadline.

## L_absent_class — Allowed class absent from GT

**Question:** Do predictions for a class absent from the sequence affect mean AP?

**Hypothesis:** Absent allowed classes may be excluded from evaluated categories.

**Setup:** One-frame synthetic scene with only hangar GT. Add a confidence-1 tank FP, or send only that FP.

**Exact command (complete suite, JSON key `L_absent_class`):**

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

**Expected from static analysis:** Hangar oracle AP remains 1 when tank FP is added; tank is absent from returned per-class keys.

**Observed:**

| Case | AP | Evaluated classes |
| --- | --- | --- |
| oracle | 1.000000 | hangar |
| oracle_plus_absent_class | 1.000000 | hangar |
| absent_class_only | 0.000000 | hangar |

**Matched expectation?** Confirmed in this synthetic setup.

**Interpretation:** The absent allowed class is not included in the evaluated category set; sending only that absent-class prediction does not rescue the missing hangar.

**Possible implication:** Interpret per-class means using the GT-present category set.

**Remaining uncertainty:** Unknown hosted category presence prevents safely relying on this behavior operationally.

# Static-analysis corrections

1. **Local realtime is not a paced 3 fps emitter.** It uses the 1/3-second clock to skip when behind, but does not wait when ahead. The synthetic timing control measures early requests. This observation does not establish hosted scheduler behavior.
2. **333 ms is not an endpoint-only safe threshold.** Image preparation is inside elapsed sequence time and outside RTT. On this host it already creates skips at zero endpoint sleep.
3. **A “high-ranked duplicate” is not automatically an FP.** The first identical matching box can claim GT. All single-object duplicate variants scored 1; the two-object test isolates the rank-dependent penalty.
4. **FPs need not reduce interpolated AP.** In C, a trailing FP has zero AP cost; the same FP ranked first halves AP. The README warning is useful but not an unconditional metric law.
5. **Wrong request_id/frame are context-invalid, not schema-invalid.** They pass the DTO yet lose the whole response at replay validation.
6. **Missing-frame fraction is not the AP loss fraction.** Visibility differs by class, the metric macro-averages classes, and COCO uses 101 recall thresholds.

No authoritative evaluator fix was made. The task measures its executable behavior; any decision to change local pacing requires separate review against the hosted implementation.

# Confirmed evaluator behaviors

- Supplied oracle: 1.000 overall and in all 16 classes.
- Correct global predictions outside the transmitted crop count equally in the tested scene.
- IoU .50 is inclusive; .499 fails; .60 and .90 produce equal AP in the one-object fixture.
- Confidence order changes AP; squaring scores without changing order preserves it.
- Duplicate AP cost depends on remaining useful detections and ordering.
- Unpredicted frames retain GT and reduce recall/AP.
- Actual latency sweeps follow the local elapsed-time recurrence; overhead and early sending matter.
- The reported COCO configuration drops the useful same-class detection at rank 101.
- Tested invalid responses lose good detections and suppress camera-command processing.
- A schema-valid out-of-bounds camera command preserves detections, keeps the old view, and generates feedback.
- Feedback repeats until a successful command; the following request first reflects the clearing.
- 3000 ms endpoint sleeps were accepted; 3500 ms sleeps timed out in this environment.
- An allowed class absent from synthetic GT was excluded from per-class evaluation.

# Scorer Version Parity — 1.7.2 vs 1.8.0

Follow-up UTC: 2026-09-17T22:26:44.362129+00:00. **All numeric results identical: True.** Compared 29 deterministic cases and 148 overall/per-class AP values using exact equality, without rounding or tolerance. Also compared maxDets/useCats/recall thresholds, independently calculated IoUs, and fixture/recall metadata.

| Probe | Cases | All AP and per-class AP identical |
| --- | --- | --- |
| Oracle | 1 | True |
| A | 2 | True |
| B | 5 | True |
| C | 4 | True |
| D | 6 | True |
| E | 5 | True |
| G | 3 | True |
| L | 3 | True |

| Requested scorer | Verified imported version | Loaded module |
| --- | --- | --- |
| 1.7.2 | 1.7.2 | C:\Users\mbnas\.vscode\nordicAI\Gforce-ai-cup-2026\drone-flyby\evaluator_probes\.cache\scorer-1.7.2\faster_coco_eval\__init__.py |
| 1.8.0 | 1.8.0 | C:\Users\mbnas\.vscode\nordicAI\Gforce-ai-cup-2026\drone-flyby\evaluator_probes\.cache\scorer-1.8.0\faster_coco_eval\__init__.py |

Isolation: fresh subprocess per version; separately installed package directories take precedence in `sys.path`; both package metadata and loaded module path are checked before scoring. Other dependencies are identical, `.venv` remains unchanged at 1.8.0, and authoritative source/data hashes match the original suite. Each case ran once per version. **A is a score-only recheck:** identical globally converted predictions are scored with the prior in-crop/off-crop geometry recorded as metadata. Crop state is not an input to `score`; no camera movement, replay, timing, or HTTP experiments were run.

**Every difference, including tiny differences:** None. Maximum absolute numeric difference is **0**.

**Comparison with the previously recorded suite:** Both versions reproduce every recorded overall/per-class AP exactly.

**Exact commands from repository root:**

```powershell
& drone-flyby/.venv/Scripts/python.exe -m pip install --no-deps --target drone-flyby/evaluator_probes/.cache/scorer-1.7.2 faster-coco-eval==1.7.2
& drone-flyby/.venv/Scripts/python.exe -m pip install --no-deps --target drone-flyby/evaluator_probes/.cache/scorer-1.8.0 faster-coco-eval==1.8.0
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/scorer_parity.py
```

The runner executed these score-only subprocess commands once each (already included in the runner above; do not repeat them as extra runs):

```powershell
& "C:\Users\mbnas\.vscode\nordicAI\Gforce-ai-cup-2026\drone-flyby\.venv\Scripts\python.exe" "drone-flyby/evaluator_probes/scorer_parity.py" "--worker-version" "1.7.2" "--package-dir" "drone-flyby/evaluator_probes/.cache/scorer-1.7.2" "--output" "drone-flyby/evaluator_probes/.cache/parity-1.7.2.json"
& "C:\Users\mbnas\.vscode\nordicAI\Gforce-ai-cup-2026\drone-flyby\.venv\Scripts\python.exe" "drone-flyby/evaluator_probes/scorer_parity.py" "--worker-version" "1.8.0" "--package-dir" "drone-flyby/evaluator_probes/.cache/scorer-1.8.0" "--output" "drone-flyby/evaluator_probes/.cache/parity-1.8.0.json"
```

**Strategic conclusions changed:** False. Oracle, inclusive IoU .50 matching, confidence ordering, duplicate ranking, missing-frame loss, rank-100/rank-101 truncation, and absent-class behavior are unchanged. **Scorer-version parity is confirmed for these fixtures, and the evaluator-probe phase can be closed.** This does not assert equivalence for untested inputs or hosted timing/network behavior.

# Remaining uncertainties

Hosted scorer version, network behavior, preprocessing costs, and scheduler parity are unverified. No hosted validation/evaluation attempt was spent. Local deterministic scoring was compared under 1.7.2 and 1.8.0 as recorded in the parity section; this does not cover untested inputs or every permitted 1.x release. Helsinki contains 25 frames and cannot establish long-run boundary stability or hosted-scene class weighting. No confidence ties, ambiguous multi-GT assignments, all camera rejection branches, or feedback across skipped frames were experimentally exhausted. Bare NaN JSON is nonstandard and was not transmitted; numeric NaN was tested directly against the DTO, while legal strings and numeric overflow were tested through HTTP. Timing values are single-run observations, not confidence intervals or portable latency guarantees.

# Implications

Capability requirements supported by these probes are whole-frame coordinate consistency, useful confidence ordering, preserving frame opportunities, sufficient within-class rank, strict response integrity, correct request identity, and explicit camera-feedback handling. Latency assessment must include local image preparation and report actual skipped/unanswered frames. No detector, tracker, architecture, or camera-control algorithm is recommended here.

# Next phase

**Yes: local evaluator behavior is sufficiently verified to proceed to Dataset & Scene Audit.** Carry forward the measured ranking semantics and the local clock/preprocessing caveat. This is readiness for a dataset audit, not certification of hosted timing or solution performance.

# Reproducibility

See `README.md` for fresh-clone setup on Windows and Linux/macOS. The exact installed versions are in `requirements.lock.txt`; all satisfy the supplied requirements. From repository root:

```powershell
py -3.12 -m venv drone-flyby/.venv
& drone-flyby/.venv/Scripts/python.exe -m pip install -r drone-flyby/evaluator_probes/requirements.lock.txt
& drone-flyby/.venv/Scripts/python.exe drone-flyby/local_evaluator.py --oracle
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

The complete command writes `probe_results.json` and regenerates this report. `--output other.json` preserves a separate run and writes `other.md`. To render an existing default JSON again:

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/report.py
```

Allow roughly 4-7 minutes on this host, with additional variation from I/O and scheduling. The server binds loopback ephemeral ports and is shut down after each replay. Temporary fixture images/annotations are automatically removed; environments and bytecode remain ignored. No large logs, images, models, or generated binaries are committed. Existing ignore rules suffice. Exact AP should reproduce under the locked scorer; exact timing and skipped-frame counts should not be expected to match across machines.

**Conclusion audit:** Every strong metric/protocol claim above has an exercised scorer or HTTP replay case. The report distinguishes direct measurements, COCO interpolation interpretation, and limitations; it does not promote local observations to hosted guarantees. Source/data hashes are checked before and after the suite.
