# Frozen V2 paired background-transfer replay

## Executive result

**CASE C — BACKGROUND HYPOTHESIS WEAK OR NOT ISOLATED STRONGLY ENOUGH.** Hosted scenery did not degrade the identical inserted patches: aggregate Recall@20 at IoU 0.30 was higher, top-20 gains exceeded losses, and paired rank/score uncertainty included no change. The observed hosted failure therefore is not explained mainly by wider scenery context in this controlled probe.

This is a controlled compositing result, not a measurement of hidden Validation recall. There is no hosted ground truth, and hosted captures are used only as scenery evidence.

## Experiment identity

- Branch: `drone-v3-background-transfer-probe`
- Experiment input HEAD: `c89b562aa51be4387349cd38cd995ff7da57ec4f`
- Frozen V2 lineage: `6fb764e5b9fb5f7eefcb832c72b35553165219e5`
- Random seed: `20260919`
- Paired examples: **96** (192 composites)
- Source Helsinki targets: **8**, one selected view for each of eight physical classes
- Hosted backgrounds: **4**, time-stratified across scene types
- Scale-bin counts: `{'8to16': 32, 'gt16': 32, 'lt8': 32}`
- Runtime: OpenCV `5.0.0`, NumPy `2.5.3`, ONNX Runtime `1.30.0` on `CPUExecutionProvider`

## Primary recall metrics

| Background | Scale | R@1 IoU .30 | R@5 IoU .30 | R@20 IoU .30 | R@1 IoU .50 | R@5 IoU .50 | R@20 IoU .50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helsinki | lt8 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| helsinki | 8to16 | 0.0% | 0.0% | 15.6% | 0.0% | 0.0% | 6.2% |
| helsinki | gt16 | 0.0% | 0.0% | 15.6% | 0.0% | 0.0% | 15.6% |
| helsinki | all | 0.0% | 0.0% | 10.4% | 0.0% | 0.0% | 7.3% |
| hosted_validation | lt8 | 0.0% | 3.1% | 6.2% | 0.0% | 0.0% | 0.0% |
| hosted_validation | 8to16 | 6.2% | 40.6% | 59.4% | 3.1% | 12.5% | 18.8% |
| hosted_validation | gt16 | 0.0% | 25.0% | 37.5% | 0.0% | 21.9% | 28.1% |
| hosted_validation | all | 2.1% | 22.9% | 34.4% | 1.0% | 11.5% | 15.6% |

## Paired result

Deltas are hosted-minus-Helsinki for the exact same target patch and geometry. Rank uses the first score-ranked proposal at IoU >= 0.30; a missing match is assigned one past the larger proposal count in that pair.

- Target-rank delta: median **0.00**, mean **-23.70** (paired bootstrap 95% CI **-235.29 to 176.94**). Positive is worse on hosted scenery.
- Target-score delta: median **0.000000**, mean **-0.001765** (95% CI **-0.009435 to 0.003279**).
- Target-score ratio (hosted/Helsinki): median **0.9037633628404199**.
- Best-IoU delta: median **-0.0180**, mean **0.0021** (95% CI **-0.0342 to 0.0398**).
- Crossed out of top-20 at IoU >= 0.30: **5.2%**; crossed into top-20: **29.2%**.
- Crossed below production threshold 0.01: **1.0%**; crossed above: **5.2%**.

**Control-sensitivity caveat:** Helsinki-composite Recall@20 at IoU 0.30 was only 10.4%. The synthetic copy-paste control therefore does not recreate a high-rank local baseline. Case C is supported by the absence—and reversal—of a hosted-background degradation, but the low control sensitivity limits how strongly this probe can exclude subtler context effects.

## Diagnostic distributions

All values are p10 / median / p90. Target score is the highest-ranked match at IoU >= 0.30; matched rank is conditional on such a match existing. Background proposals mean candidates below IoU 0.30 relative to the inserted target, not labeled negatives.

| Background | Scale | Matched rank median / p90 | Target score | Best target IoU | Proposals before top-K | Background/non-target proposals |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| helsinki | lt8 | 1210.0 / 1751.0 | 0.000000 / 0.000000 / 0.000210 | 0.009 / 0.241 / 0.377 | 1548.1 / 2030.5 / 2343.5 | 1548.1 / 2029.5 / 2343.5 |
| helsinki | 8to16 | 84.5 / 832.5 | 0.000000 / 0.000611 / 0.004631 | 0.098 / 0.407 / 0.618 | 1559.7 / 2030.5 / 2344.6 | 1559.6 / 2029.5 / 2343.6 |
| helsinki | gt16 | 130.5 / 392.6 | 0.000452 / 0.001038 / 0.003652 | 0.558 / 0.663 / 0.777 | 1579.2 / 2054.0 / 2358.1 | 1574.5 / 2050.0 / 2355.1 |
| helsinki | all | 262.0 / 1221.2000000000003 | 0.000000 / 0.000444 / 0.003006 | 0.069 / 0.408 / 0.701 | 1552.5 / 2043.5 / 2346.0 | 1552.0 / 2041.0 / 2346.0 |
| hosted_validation | lt8 | 12.0 / 26.400000000000002 | 0.000000 / 0.000000 / 0.000000 | 0.102 / 0.229 / 0.297 | 2341.2 / 2898.0 / 3334.6 | 2340.3 / 2898.0 / 3334.6 |
| hosted_validation | 8to16 | 12.0 / 1218.7 | 0.000157 / 0.001398 / 0.013880 | 0.332 / 0.443 / 0.586 | 2343.4 / 2894.0 / 3336.3 | 2342.2 / 2892.0 / 3333.4 |
| hosted_validation | gt16 | 167.0 / 324.90000000000003 | 0.000470 / 0.000906 / 0.002642 | 0.472 / 0.598 / 0.685 | 2360.6 / 2922.5 / 3337.0 | 2357.5 / 2916.5 / 3333.0 |
| hosted_validation | all | 20.0 / 711.2 | 0.000000 / 0.000620 / 0.004249 | 0.197 / 0.442 / 0.669 | 2345.0 / 2915.5 / 3337.0 | 2343.5 / 2911.5 / 3334.5 |

Full rank, score, IoU, proposal-volume, background-proposal, boundary-crossing, scale, class, and background records are in `pair_metrics.csv`; every post-NMS diagnostic candidate is in `candidate_metrics.csv`.

## Rendering and controls

- Helsinki scenery stays at its native 3840×2160 source resolution.
- Each hosted 960×540 capture is expanded to 3840×2160 with nearest-neighbor replication. The subsequent 4× `INTER_AREA` render therefore preserves the original hosted scenery pixels outside the pasted rectangle.
- A tight annotated Helsinki bbox is cropped, resized at source resolution, and pasted at a 4-source-pixel-aligned location. The complete 3840×2160 composite is then rendered to 960×540 with `cv2.INTER_AREA`.
- The same resized patch bytes, x/y location, zero rotation, bbox geometry, and render path are used in each pair. A byte equality assertion verifies the final transmitted target rectangle in every pair.
- Placement is selected from a fixed grid among low-gradient regions in both backgrounds and excludes known Helsinki annotation boxes. This reduces unnecessary overwriting of salient structures; it does not assert that hosted regions are target-free.
- The frozen ONNX head is decoded at `1e-4`, with V2's class pooling and IoU 0.55 NMS, but without its confidence 0.01 floor or 32-proposal truncation.

## Bounding-box caveat

The pasted target is a tight rectangular annotation crop, not a segmentation cutout. It can contain original Helsinki background pixels inside the bbox. Because that exact rectangle is reused byte-for-byte within each pair, the experiment isolates sensitivity to surrounding/wider scenery context, not pure foreground/background isolation.

Hosted images are not negative labels. They may contain hidden challenge targets, so all non-matching candidates are called background/non-target *relative to the inserted synthetic target only*.

## Interpretation gate

The gate was fixed in code before interpretation: Case A requires Helsinki R@20 >= 0.50, at least 80% R@20 retention, median score ratio <= 0.5, and median rank degradation <= 5. Case B requires Helsinki R@20 >= 0.50 plus either <=70% R@20 retention or >=20% paired top-20 losses. All other outcomes are Case C.

**Final classification: CASE C — background hypothesis weak or not isolated strongly enough.**

## Limitations

- No hosted ground truth exists; nothing here estimates actual hidden Validation recall, target prevalence, target scale, or class accuracy.
- Hosted backgrounds originate as transmitted 960×540 captures. Nearest-neighbor expansion makes source-resolution compositing deterministic but cannot reconstruct source detail that was never captured.
- Rectangular crops retain some local Helsinki context inside each bbox.
- Repeated use of one physical Helsinki instance per class avoids adjacent-frame inflation but does not test new physical-instance generalization.
- Existing unlabeled objects in either background can compete in rank; paired analysis and four diverse scenes expose rather than label that competition.

## Exactly one recommended next action

**Run one paired physical-instance-transfer probe that holds scale and scenery fixed while swapping Helsinki target instances for genuinely independent target instances.**

## Reproduction

```powershell
& 'drone-flyby/.venv/Scripts/python.exe' 'drone-flyby/architecture_experiments/v3_background_transfer_probe/run_background_transfer_probe.py'
```

The script fails closed on branch mismatch, artifact hash drift, missing/wrong-sized inputs, pair pixel inequality, unbalanced bins, or inconsistent row counts. Generated composites are reproducible in the ignored `cache/` directory when `--save-composites` is supplied.
