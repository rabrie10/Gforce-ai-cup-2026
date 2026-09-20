# DECISION (from the 20-image hosted L1/L2 audit)

**What is failing.** Mainly ms1 *raw discovery on hosted imagery*: 8 of 14 reliable visible targets (incl. 2 EASY L2 tanks, f6/f7) get no useful raw proposal. A second, independent layer of cheap pipeline gates kills 4 more targets that *did* have a proposal: production `det_conf=0.15` (f52 helicopter, raw top-1 at .112), `classify_min_px=22` (f101 21.8 px, f105 17.5 px → never classified → never emitted), and the bg head threshold (f204 large launcher, target_p .48). Merge/NMS lost nothing. Coordinate conversion shows no error.

**Certainty.** High that the replay equals production (exact candidate reproduction on 20/20 frames). Moderate that raw discovery is the dominant failure. Low on precise rates (14 targets ≈ 9 independent objects, my own boxes, unknown GT box convention, unknown classes). Tracking could not be audited (no per-track state, 20/228 frames).

**Are easy objects missed?** Yes: 49x26 px tank at native L2 (f6/f7) and helicopters (weak/threshold-lost). ms1 succeeded on objects over bare sand/dirt and missed similar ones over grass/forest edges.

**Detection vs later stages.** Both, but ordered: raw detection (8/14) > gates (4/14) > class quality. Because the raw stage dominates, tuning gates alone can recover at most ~4/14.

## Recommended single next path: PATH 1 — replace/augment discovery (not tune the pipeline first)

Step 0 (≈30–45 min, gating, no integration): on the **same frozen 14 annotations / 20 frames**, compare raw-proposal recall (best IoU over all, top-16/32/64, rank, latency) of ms1 vs yolo11l-obb vs YOLO-World vs GLEE-Lite (already installed) at L1/L2 — earlier comparisons were L0 only. If any source gets raw IoU≥0.3 on most of the EASY/MODERATE targets (esp. f6/f7, f4, f50/f51) in its top-32, adopt it as an extra proposal source.
Step 1 (only if Step 0 fails): retrain the one-class discovery detector with a physically-matched generator: hosted frames are the *same 3D models composited into aerial imagery*, so cut-out-based copy-paste of the 16 Helsinki objects (mask-based, not box feathering as in `composite_dataset.py`) with full 360° rotation, L0/L1/L2 scale sampling, and many diverse background crops (Helsinki background-only regions + unlabeled non-hosted aerial imagery), plus hard negatives; init from ms1 or an aerial-pretrained detector; keep discovery one-class and class recognition separate; validate **only** on the frozen hosted annotations (never train on the 20 diagnostic images).

Riders (cheap, do after Step 0, not competing paths): lower `det_conf` (~0.05 with top-k), drop/rethink the 22 px classify gate for L1, review the 0.5 bg threshold, and stop emitting `condor`/`small_tower` low-confidence spam.

**Not recommended:** spending the remaining time on NMS/tracker tuning, another hosted Validation, or a long training run before Step 0.
