# Architecture V2 hosted zero-score forensic audit

## Executive finding

**[CONFIRMED] The largest hosted-versus-Helsinki change occurred at discovery.** Hosted Validation produced 0.754 proposals/frame, versus 8.000 in the matched Helsinki run, and 121 of 248 observed hosted frames (48.8%) had no proposal. The proposal-score median fell from 0.5501 to 0.0195. None of the 187 hosted proposals reached 0.5; 51% of the 200 Helsinki proposals did.

**[SUPPORTED] The surviving hosted crops were also much less class-separable.** Observation top-posterior median fell from 0.6537 to 0.2678. Track primary-posterior median fell from 0.3499 to 0.0978, and the median top-versus-runner-up posterior gap fell from 0.2227 to 0.0082. The track bank therefore remained small and weak: 1.31 tracks/frame hosted versus 9.80 locally.

**[SUPPORTED] Discovery starvation plus recognition ambiguity is sufficient to explain a severe output collapse, but not to prove why the hidden score was exactly zero.** Hosted output fell from 22.52 to 3.90 annotations/frame, and 78 of 248 recorded hosted frames (31.5%) emitted nothing. Without hosted ground truth, proposal recall, classification accuracy, IoU, and the correctness of the remaining annotations are not observable.

**[CONFIRMED] Latency, malformed-response rejection, GMC failure, and a camera-level change do not explain the measured collapse.** The organizer reported no errors. Hosted internal latency was about 103 ms mean and 116 ms p95. GMC accepted every frame after initialization, and both runs stayed at resolution level 0 with no camera request or feedback.

## Scope and evidence

- Frozen V2 lineage: commit `6fb764e5b9fb5f7eefcb832c72b35553165219e5`.
- Hosted attempt: `6a86911aebf64be79a3b6a6742de0686`, raw score 0, organizer errors `[]`.
- Hosted endpoint calls: 249. Telemetry accepted/written: 249/249; dropped/errors: 0/0.
- Hosted forensic files: 248 frame JSON records and 50 sampled images. `frames/00001.json` is missing.
- Matched Helsinki evidence: 25 telemetry records from the identical frozen candidate; the associated local evaluation remained approximately 0.431 mAP@0.50.
- Comparisons use per-frame rates and distributions because the runs have different lengths and are not frame-aligned.
- Image comparisons use the 50 hosted samples and all 25 Helsinki frames. Helsinki images were rendered at the common 960×540 transmitted size with `INTER_AREA`, matching the evaluator path.

**[CONFIRMED] The missing hosted record remains unresolved.** The Portal Test immediately preceded Validation without a container restart, and telemetry filenames use only `frame_index`, so a session collision or overwrite is plausible. It is not established as a transfer artifact. Frame 1 was not interpolated, and no warm-up conclusion depends on it.

## Headline stage comparison

| Stage metric | Hosted | Matched Helsinki | Status |
| --- | ---: | ---: | --- |
| Proposals/frame, mean | 0.754 | 8.000 | **CONFIRMED** |
| Proposals/frame, median / p95 / max | 1 / 3 / 5 | 9 / 12.6 / 13 | **CONFIRMED** |
| Zero-proposal frames | 121/248 (48.8%) | 0/25 | **CONFIRMED** |
| Proposal-score median | 0.0195 | 0.5501 | **CONFIRMED** |
| Crops/frame, mean | 0.754 | 8.000 | **CONFIRMED** |
| Observations/frame, mean | 0.714 | 7.320 | **CONFIRMED** |
| Observation top-posterior median | 0.2678 | 0.6537 | **CONFIRMED** |
| Track primary-posterior median | 0.0978 | 0.3499 | **CONFIRMED** |
| Track posterior-gap median | 0.0082 | 0.2227 | **CONFIRMED** |
| Track bank size/frame, mean | 1.31 | 9.80 | **CONFIRMED** |
| Emitted annotations/frame, mean | 3.90 | 22.52 | **CONFIRMED** |
| Zero-output frames | 78/248 (31.5%) | 0/25 | **CONFIRMED** |
| Distinct output classes/frame, mean | 2.98 | 8.68 | **CONFIRMED** |
| GMC accepted after initialization | 247/247 | 24/24 | **CONFIRMED** |
| Actual resolution level | L0 on 248/248 | L0 on 25/25 | **CONFIRMED** |

## Discovery

**[CONFIRMED] Proposal volume collapsed across most of the hosted run.** The hosted mean/median/p95/max were 0.754/1/3/5, compared with 8.000/9/12.6/13 in Helsinki. The effect was persistent, not confined to initialization. In 25-frame blocks, hosted mean proposal count ranged from 0.12 to 1.92. Frames 125–149 contained 23 zero-proposal records out of 25.

**[CONFIRMED] Detector activation strength changed more than proposal geometry.** Hosted proposal score mean/median/p95/max were 0.0296/0.0195/0.0874/0.2586. Helsinki values were 0.4555/0.5501/0.9761/0.9988. Only 2.7% of hosted proposals reached 0.1, versus 54% locally; none reached 0.5, versus 51% locally.

**[CONFIRMED] Hosted proposals were smaller and less variable, but the geometric shift was secondary to the score collapse.** Hosted versus Helsinki medians were:

- width: 16.0 versus 19.25 received pixels;
- height: 15.6 versus 19.3 pixels;
- short side: 15.0 versus 18.05 pixels;
- area: 247 versus 357 received-pixel².

The hosted short-side p95 was 18.54 pixels, versus 32.05 locally. Hosted proposal centers also shifted left/up on average (normalized center 0.419, 0.457 versus 0.627, 0.530), consistent with different scene content and trajectories. Without hosted GT, this spatial difference cannot be labeled missed-object localization.

**[NOT OBSERVABLE] Hosted proposal recall and object-size-conditioned recall cannot be computed.** There are no hosted annotations against which to match proposals.

## Recognition

**[CONFIRMED] Recognition volume was almost entirely constrained by discovery.** There were 187 hosted crops and 177 observations, versus 200 crops and 183 observations in the much shorter 25-frame Helsinki run. This equals 0.754 crops and 0.714 observations per hosted frame, versus 8.000 and 7.320 locally.

**[CONFIRMED] Background rejection was not the main loss point.** Hosted rejected 10 of 187 crops (5.3%); Helsinki rejected 17 of 200 (8.5%). Conditional per-positive-frame rejection fractions averaged 5.6% and 8.2%, respectively.

**[CONFIRMED] Objectness stayed broadly similar on frames with crops, while identity separation fell sharply.** Conditional mean objectness averaged 0.880 hosted versus 0.906 locally. Conditional mean margin averaged 0.0286 versus 0.1246. Observation top-posterior mean/median/p95 were 0.282/0.268/0.441 hosted and 0.594/0.654/0.846 locally.

**[SUPPORTED] The hosted recognizer collapsed onto a narrower and different set of top classes.** Nine classes appeared as an observation's top class hosted, versus thirteen locally. Hosted observation tops were concentrated in `jammer` (53), `ta-ta` (37), `spacecraft` (32), and `small_launcher` (21). This is observable prediction behavior, not evidence that those hidden classes were present or absent.

**[NOT OBSERVABLE] Hosted classification accuracy and hidden class-specific failure are unavailable.** The class-frequency comparison describes predictions only.

## Track state

**[CONFIRMED] Hosted tracks were fewer, less concentrated, and less persistent.** The mean bank size was 1.31 hosted versus 9.80 locally. Median primary identity posterior was 0.0978 versus 0.3499; median primary-to-runner-up gap was 0.0082 versus 0.2227. Mean confirmed/tentative tracks per frame were 0.76/0.55 hosted and 7.80/2.00 locally.

**[CONFIRMED] Creation and retention were weak relative to run length.** Hosted created 59 tracks, matched 118 updates, and removed 58 tracks over 248 recorded frames. Helsinki created 29, matched 154, and removed 24 over 25 frames. Only 39.0% of hosted track IDs ever became confirmed, versus 72.4% locally. Median observed lifecycle span was 3 frames hosted versus 5 locally; only 15.3% of hosted tracks reached five hits, versus 41.4% locally.

**[SUPPORTED] Track/output starvation is mainly downstream of sparse discovery, with recognition ambiguity reducing identity consolidation.** GMC propagation was present in both runs and track geometry confidence was similar (median 0.875 hosted versus 0.947 locally), so the main track-state divergence is not a wholesale motion-propagation failure.

## Output

**[CONFIRMED] Hosted output volume and per-frame class coverage collapsed.** Hosted emitted 3.90 annotations/frame (median 3, p95 14.65, max 21) versus 22.52 (median 27, p95 34.8, max 38) locally. Seventy-eight hosted records emitted zero annotations; Helsinki had none. No recorded frame in either run dropped an invalid annotation.

**[CONFIRMED] Secondary-guess volume fell with the track bank.** Hosted emitted 2.59 secondary guesses/frame versus 12.76 locally. Recorded distinct classes/frame averaged 2.98 hosted versus 8.68 locally. The maximum in one frame establishes only a lower bound of 11 distinct hosted output classes over the run, versus 12 locally.

**[NOT OBSERVABLE] Exact whole-run emitted class frequencies and emitted-confidence distributions are not in telemetry.** Telemetry stores per-frame output counts and track top/runner-up identities, but not the emitted annotation list, third posterior, or emitted confidence. The derived class CSV therefore labels track-primary and observation-top counts; it does not present them as emitted-class counts.

## Global motion compensation

**[CONFIRMED] GMC was healthy and closely matched.** Excluding each run's `no-previous-frame` initializer, hosted accepted 247/247 and Helsinki accepted 24/24. Accepted-frame mean quality/inlier ratio was 0.9920 hosted and 0.9946 locally; median inliers were 292 and 295. Median translation magnitude was 54.42 versus 53.81 pixels. Hosted had one 109.39-pixel maximum, while its p95 remained 55.20; the local maximum was 54.48.

**[NOT OBSERVABLE] Affine-matrix scale, rotation, and shear magnitudes were not recorded.** Telemetry records the model name, quality, inliers, reason, and translation only.

## Camera behavior

**[CONFIRMED] Both systems effectively stayed at L0.** Every observed hosted and Helsinki frame reported resolution level 0 and the full `[0, 0, 3840, 2160]` source region transmitted at 960×540. Every camera decision was `hold-current-view`; there were no requested camera moves and no camera feedback records.

This removes camera-level selection as a hosted-versus-local difference. It does not establish that L0 was adequate for hidden hosted objects.

## GT-free image characteristics

**[CONFIRMED] The sampled hosted inputs are a visibly and quantitatively different scene distribution.** The hosted contact sheet traverses open grass/road, airport, industrial/harbor, and dense urban scenes. Helsinki is one forest/lakeshore sequence. At the common input size:

- mean luma was 98.19 hosted versus 74.51 locally (+31.8%);
- luma contrast was 47.30 versus 59.80 (-20.9%);
- mean gradient magnitude was 98.20 versus 130.92 (-25.0%);
- Laplacian variance was 2,171.9 versus 3,659.2 (-40.6%);
- edge density was 0.2525 versus 0.2777 (-9.1%);
- dark-pixel fraction was 0.150 versus 0.410;
- luma entropy was similar, 7.393 versus 7.454 bits.

**[SUPPORTED] This domain shift is consistent with the detector-score collapse.** Brighter, lower-contrast, lower-high-frequency scenes can alter small-object proposal activations, and the sampled content differs substantially from the Helsinki reference scene. This correlation does not prove object scale, hidden object presence, or proposal recall.

**[NOT OBSERVABLE] Apparent hidden-object scale is not measurable without hosted object locations.** Proposal-size distributions are measured, but proposal size must not be substituted for ground-truth object size.

## Ranked failure taxonomy

1. **Discovery activation collapse — CONFIRMED divergence; SUPPORTED zero-score mechanism.** Proposal count fell 10.6×, median score fell 28.2×, 48.8% of recorded frames had no proposals, and no hosted proposal reached 0.5. This is the strongest measured upstream change.
2. **Recognition identity ambiguity on surviving crops — CONFIRMED divergence; SUPPORTED compounding mechanism.** Top posterior, margin, and posterior gap all fell substantially, and top-class behavior narrowed/shifted.
3. **Sparse, short-lived track bank and output starvation — CONFIRMED divergence; SUPPORTED downstream mechanism.** Hosted bank size was 13.4% of local, emissions were 17.3%, and 31.5% of recorded frames emitted nothing.
4. **Hosted visual-domain shift — CONFIRMED input difference; SUPPORTED upstream driver.** Sampled hosted scenes are brighter and less locally contrasted/textured than Helsinki and show much broader terrain/urban content. Causality is not yet isolated from runtime/model/config drift.
5. **Telemetry session/file collision — PLAUSIBLE for the missing frame only.** The prior Portal Test and frame-index-only filenames make collision plausible, but this does not explain the run-wide prediction collapse and is not confirmed.

**[CONFIRMED exclusions]** Slow inference, response errors, widespread GMC rejection, and a changed camera level are contradicted by the available evidence.

## Exactly one recommended next experiment

**Run a deterministic shadow replay of the 50 captured hosted 960×540 images through the frozen V2 discovery and recognition stages, recording model/config/runtime hashes and comparing proposal boxes/scores and recognition posteriors exactly against the hosted telemetry.** Process each image independently for discovery, and feed the recorded proposal boxes to recognition so stateful frame gaps do not confound the comparison. Exact agreement would isolate the measured collapse to the hosted input distribution; disagreement would identify environment, artifact, configuration, or preprocessing drift. This is the highest-information diagnostic because it separates the two leading causal families using the exact hosted input bytes, without hosted GT, retraining, architecture changes, deployment, or another validation.

## Limitations

- No hosted ground truth exists. Hosted proposal recall, classification accuracy, IoU, object scale, and class-specific AP remain **NOT OBSERVABLE**.
- Hosted `frame_index=1` is missing and was neither interpolated nor used for a warm-up claim.
- The 50 hosted images are a stride-5 sample, not all 249 endpoint inputs.
- Helsinki has 25 frames from one scene; hosted has 249 endpoint calls across a broader trajectory. Distribution comparisons are valid, but the runs are not paired frame-for-frame.
- Exact emitted class IDs/confidences and full affine matrices were not recorded.
- The analysis attributes measured internal changes. It cannot prove the exact condition that made hidden mAP numerically zero.

## Reproduction

From the repository root:

```powershell
& 'drone-flyby\.venv\Scripts\python.exe' 'drone-flyby/architecture_experiments/v2_hosted_failure_audit/analyze_audit.py'
```

The script reads raw telemetry/images in place and writes only derived CSV, JSON, and PNG files under this audit directory. Raw hosted telemetry and images are not part of the audit commit.
