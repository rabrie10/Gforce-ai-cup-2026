# V6 second hosted Validation forensics (attempt 424a4205, mAP 0.0127; run1 was 0.0178)

## Reconstruction (from preserved endpoint_v6_public_9053.log.run2_424a4205_*)
- 185 evaluator /predict received (frame_index 0-248; 64 skipped by the realtime 3fps clock).
- Camera levels used: L1=101, L2=83, L0=1 (multiscale + active camera working; NO illegal commands).
- Server duration_ms: p50 52, p95 67, max 175. Predictions/frame: median 4, max 13, total 822.
- No exceptions, CUDA errors, or restarts.

## The two "timeouts" are NOT server-side
- Frame 128 (fi127): server answered in 73.4 ms, status ok.  Frame 238 (fi237): 45.4 ms, status ok.
- No server duration exceeded 3333 ms (or even 1000 ms). The >3333 ms was network / Runpod HTTP-proxy
  round-trip latency (2 of 249 frames), consistent with the earlier honest proxy-latency caveat.
  Not a compute bottleneck.

## Real bottleneck: detector domain generalization (measured)
The pipeline emits (822 predictions) but mAP ~= 0. Direct detector test on hosted-domain captures
(hosted_detector_compare.py, frames 60/125 vs the manual boxes):
- ms1 (Helsinki fine-tuned V6 detector): **best IoU 0.00 on ALL 8 hosted targets** (full & tiled),
  yet emits 13-28 background proposals/frame -> all false positives.
- OBB + YOLO-World (pretrained, general): the ONLY source with any hosted recall (heli 0.37, plane
  0.12) but still < 0.50 on every target.
So the near-zero hosted mAP is the DETECTOR failing to localize hosted objects to IoU>=0.50; the
camera fix and latency are not the cause. Both hosted runs (~0.018, ~0.013) are effectively zero;
their difference is noise.

## Targeted change implemented + measured: ensemble discovery (env-gated, DEFAULT OFF)
Added ms1 + OBB + YOLO-World union (NMS dedup) as `V6_ENSEMBLE=1` (pipeline_v6._detect). Measured:
- Adds the only nonzero hosted proposals (heli 0.37) where ms1 gives 0.
- BUT drops local Helsinki mAP 0.265 -> 0.206 (OBB/World add competing proposals), and still does not
  reach IoU>=0.50 on hosted -> would not move hosted mAP@0.50.
Per "keep only if measured results justify it", the ensemble is **NOT** made the default. It ships as
an opt-in (V6_ENSEMBLE=1) for a possible hosted A/B; the committed default remains ms1-only
(camera-fixed, Helsinki 0.265). Camera regression tests still ALL PASS; latency 76 ms median.

## Conclusion / remaining limitation
No ~60-min change closes the hosted domain gap: every available detector (ms1, OBB, World) fails to
localize the hosted objects to IoU>=0.50. The required fix is hosted-representative detector training
data (or the actual validation-scene imagery, which is not persisted) - out of scope for a 60-min
pass and explicitly excluded (no new Helsinki-only cycle, no Codex integration). Recommend a hosted
A/B of V6_ENSEMBLE=1 only if a cheap experiment is wanted; otherwise prioritize obtaining hosted-domain
labeled data before further hosted attempts.
