# Hosted L1/L2 perception failure audit (attempt d09922e7…, sequence 369d011d…)

Branch `claude/hosted-l1l2-audit` (off `codex/v6-diagnostic-capture` 8ae3b97). Nothing on port 9053 was touched; no Validation/Final was run; no training was done.

## What was done
1. Verified all preserved artifacts against `SHA256SUMS` (all OK; hashes use the Runpod absolute paths, remapped to basenames).
2. Viewed all 20 original 960x540 images; measured boxes on 5-8x zoom crops; froze them in `manual_annotations_frozen.json` (sha256 `0dc88cb6…`, commit 370045e) **before** any replay. Disclosure: one f204 post-merge candidate line was printed in a metadata dump *after* the f204 boxes were measured (they were not changed).
3. Replayed the exact ms1 checkpoint (sha `e5172a37…`) on the 20 saved observations on the Runpod 4090 with production preprocessing (cv2 BGR PNG, `YOLO.predict(imgsz=960)`), only `conf` lowered to 0.001. **Validity check:** at production `conf=0.15` + `_merge_props(0.6)` the replay reproduces the stored `post_merge_detector_candidates` exactly (min IoU 1.0 on all 20 frames), so the replay is the production detector.
4. Ran the production `RejectingExpert` (DINOv2 + bg head) on (a) the best raw proposal, (b) the best stored post-merge candidate, (c) my manual box.

Files: `failure_matrix.csv`, `summary_counts.json`, `raw_replay_results.json`, `contact_sheet_1..5_*.png`, `overlays_audit/`, scripts `freeze_manual_annotations.py`, `raw_replay.py`, `build_audit.py`.

## Correction to the previous session
The earlier annotation "hosted_f55_helicopter [335,267,435,355]" is the **frame 52** helicopter (f52: rotors span x334-430,y275-350). The frame 55 helicopter is at the bottom edge, [287,475,382,540] (clipped). The previous "no overlapping candidate at frame 55" claim was made against the wrong location.

## What is visible (14 targets; 20 images)
* Wide, obvious synthetic-looking military objects appear in 9 of 20 images: helicopter (f52, f55), olive/camouflaged tank-like vehicles (f4, f6, f7, f50, f51, f100, f101, f105, f204), a large launcher-like truck (f204). The same vehicle recurs across f100/f101/f105 and f6/f7 and f50/f51.
* No plausible target in f5, f102, f150, f151, f153, f159, f200, f201, f203 (urban/industrial/residential/forest; cars, boats, roofs are not counted).
* Uncertain, not boxed: lattice/pylon structures in f52 (possible "tower"), equipment yard in f101.
* Difficulty (my judgment): EASY 6, MODERATE 7, HARD 1. Existence confidence: HIGH 7, MEDIUM 6, LOW 1. Classes of the vehicles are **unknown** (tank-like); do not read the class tables as ground truth.

## Failure matrix (see `failure_matrix.csv`; IoU vs my tight boxes)
| target | L | size px | raw best IoU (tight / 1.3x-padded) | raw score, rank | post-merge IoU | bg target_p on cand | final | primary failure |
|---|---|---|---|---|---|---|---|---|
| f52 heli EASY | 2 | 96x75 | .43 / .27 | **.112, rank 0** | none (0 cands) | .91 (helicopter .59) | none | D: removed by `det_conf=0.15` |
| f55 heli MOD (clipped) | 2 | 95x65 | .41 / .27 | .008, rank 25 | .00 | .99 (helicopter .84) | none | B (weak raw) |
| f6 tank EASY | 2 | 49x26 | .07 / .11 | – | .00 | (.92 on GT box) | none | B raw miss |
| f7 tank EASY | 2 | 50x26 | .05 / .09 | – | .00 | (.91 on GT box) | none | B raw miss |
| f100 tank EASY | 2 | 58x26 | .34 / .56 | .744, rank 0 | .34 | .64 (large_launcher .27) | IoU .34 | C/E box loose + class unclear |
| f204 veh EASY | 2 | 43x41 | .51 / .85 | .861, rank 0 | .51 | .84 | IoU .51 (small_launcher .18) | OK detected (class unverified) |
| f204 launcher EASY | 2 | 102x92 | .68 / .84 | .016 (cand box score .47-.50) | .60 | **.48** | none | E: bg head 0.48 < 0.5 |
| f4 tank MOD | 1 | 24x13 | .01 | – | .00 | (.80 GT box) | none | B raw miss |
| f50 tank MOD | 1 | 20x20 | .00 | – | .00 | (.96 GT box) | none | B raw miss |
| f50 dark_b MOD | 1 | 26x25 | .00 | – | .00 | (.50 GT box) | none | B raw miss |
| f50 dark_a HARD/LOW | 1 | 13x10 | .00 | – | .00 | (.97) | none | B raw miss |
| f51 tank MOD | 1 | 20x20 | .00 | – | .00 | (.99 GT box) | none | B raw miss |
| f101 tank MOD | 1 | 29x13 | .45 / .74 | .698, rank 0 | .45 | .61 | none | E: min-side gate (cand 21.8 px < 22) |
| f105 tank MOD | 1 | 30x14 | .61 / .84 | .287, rank 3 | .61 | .83 | none (0 preds in frame) | E: min-side gate (17.5 px < 22) |

**Box-convention caveat.** Helsinki GT boxes are visibly looser than the tight object extents for vehicles (padding), and about rotor-span for the helicopter. My tight boxes therefore understate ms1 for vehicles; the padded column is a sensitivity check, not truth. Localization (category C) is **not** reliably measurable from this sample; I did not use it as a primary conclusion.

## Aggregates
* By difficulty: EASY 6 = {2 raw miss, 1 threshold, 1 bg reject, 1 loose box/class, 1 detected}; MODERATE 7 = {4 raw miss, 1 weak raw, 2 min-side gate}; HARD 1 = raw miss.
* By stage (primary): raw miss/weak 8 (B), threshold 1 (D), classification/background/gate 3 (E), loose box+class 1, detected 1. Merge/NMS loss: **0**. Coordinate conversion (G): no error found (post-merge source boxes reproduce; not a cause). Camera coverage (A): every target listed was visible in the delivered view. Tracking (F): **not determinable** — per-track state was not persisted and only 20 of 228 frames exist; f101/f105 are explained structurally by the classify gate.
* Raw ms1 found a proposal at IoU>=0.3 (tight or padded) for 7/14: short side >=30 px: 4/4; 16-30 px: 1/6; <16 px: 2/4. Not a clean size effect (13-14 px-high L1 vehicles f101/f105 were found; 20-26 px f50/f51/f6/f7 were not).
* Raw score ≥0.15 and IoU≥0.3 (tight): only f100 (.74), f204 small (.86), f101 (.70), f105 (.29): 4/14.

## Answers
1. **Does ms1 miss visually obvious L2 objects? Yes.** f6/f7 (EASY, 49x26 px tank on open grass) have no raw proposal within IoU 0.11 at any confidence (ms1 emits only huge diffuse boxes over trees/grass). f52/f55 helicopters are proposed only weakly (.11 / .008) with a body-only box.
2. **Does it succeed on simple/large targets and fail on tiny/specialized ones?** Only partly. It succeeded on f204 (both L2 objects on bare dirt/sand) and f100 (sand path), but missed similar-size tanks on grass/forest edges. Hypothesis (untested): objectness is context/background-dependent rather than shape-driven.
3. **Size correlation:** weak (see above).
4. **Family:** helicopter and large launcher are recognised well by the class/bg heads (helicopter .58/.85, launcher .71) when a box exists; the tank-like vehicles get `jammer`/`small_tower`/`large_launcher` (class unknowable here, but the class head is not confident: 12/14 "tentative").
5. **Main problem: several independent issues** — dominant is raw discovery recall on hosted backgrounds (8/14, incl. EASY ones), plus three cheap, independent gate problems (det_conf 0.15, `classify_min_px=22`, bg threshold 0.5 — together they killed 4/14 targets that had real proposals: f52, f101, f105, f204-launcher), plus a weak class head on vehicles. Emission of near-everything as `condor`/`small_tower` at 0.1-0.4 confidence in non-target frames is a further precision issue.
6. **Are useful raw proposals lost by merge/NMS?** No. The lost proposals are lost to the confidence threshold (f52) and downstream gates; merge NMS (0.6) removed nothing relevant.
7. **Does genuine L1/L2 help vs earlier L0?** Suggestive only: earlier L0 audits had 0/5 firm targets at IoU≥0.5; here 3/14 raw at IoU≥0.5 (tight) and 7/14 at ≥0.3. Different targets, tiny n; not a controlled comparison.
8. **Consistent with overfit to one instance/scene?** Consistent, not proven. The hosted objects are the same 3D models (helicopter/launcher are recognised at .58–.85), so shape novelty is not the obvious cause; failure on visually similar objects against new backgrounds points to background/scale/orientation/context generalisation. ms1 training had `copy_paste=0`, `degrees=12`, and only Helsinki backgrounds.

## Limits
14 targets from 20 images of one sequence (7 L2, 7 L1), several of them the same physical object in consecutive frames (f6/f7, f50/f51, f100/f101/f105) → effectively ~9 independent objects. Manual boxes are mine, not GT. Class labels unknown. Numbers are proportions with very wide uncertainty.
