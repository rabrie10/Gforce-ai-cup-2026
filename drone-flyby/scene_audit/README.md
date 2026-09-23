# Helsinki Dataset & Scene Audit

Run from the repository root (PowerShell):

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/scene_audit/run_audit.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/scene_audit/summarize_audit.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/scene_audit/validate_audit.py
```

The existing project environment is sufficient. For a new environment, install
`drone-flyby/requirements.txt`; the audit imports the official renderer and its
dependencies. No extra audit dependencies, network calls, model downloads, training,
server, or official evaluation attempts are needed. Python 3.12 was used. Exact
Python, NumPy and OpenCV versions and input SHA-256 hashes are in `scene_audit.json`.

Read **[DATASET_SCENE_AUDIT.md](DATASET_SCENE_AUDIT.md)** first, then the generated
**[MEASUREMENTS.md](MEASUREMENTS.md)**. The first command regenerates tables and PNGs;
the second regenerates supplementary diagnostics and the measurement appendix.
The third independently checks saved IoUs using center/size arithmetic and validates
row completeness, denominators, gallery coverage and unchanged input hashes; it
writes `validation_results.json`.
The human visual judgments in the main report are intentionally not automatically
regenerated: changing inputs requires renewed pixel inspection and interpretation.

## Outputs and definitions

* `annotation_metrics.csv`: all 259 class/frame rows, source xyxy, dimensions,
  normalized dimensions, area, aspect ratio, centers, four edge distances and
  apparent L0/L1/L2 dimensions. `boundary_touch` flags potentially clipped boxes;
  it is not an occlusion annotation. Boundary means <=1 source pixel from an edge:
  this dataset clips right/bottom coordinates to 3839/2159. Width/height still use
  x2-x1/y2-y1, matching the evaluator, with no added inclusive pixel.
* `class_summary.csv`: all numeric properties, per class and pooled, with count,
  min, interpolated p10, median, mean, p90 and max. Repeated frames are correlated.
* `frame_inventory.csv`: file pairing, per-frame counts, dimensions and recorded pose.
* `scale_buckets.csv`: level/class buckets and exact class:frame membership. Buckets
  partition short side as `<4`, `[4,8)`, `[8,16)`, `[16,32]`, `>32` pixels.
* `temporal_metrics.csv`: first/last annotation, gaps, edge evidence, observed
  duration `N/3` and first-to-last timestamp span `(last-first)/3`. Frame 0 entries
  and frame 24 exits are censored; disappearance is not proof of physical departure.
* `motion_metrics.csv`, `motion_summary.csv`: consecutive source-frame changes;
  normalized displacement is Euclidean distance in `(x/W,y/H)`, scale ratio is
  square root of area ratio. Class correspondence is justified by metadata's one
  physical instance per class, not by class names alone.
* `stale_box_iou.csv`: every available future annotated horizon, hold and two-GT-box
  linear center/size extrapolation. Clipping to frame bounds happens after prediction;
  collapsed/inverted predictions count as IoU zero. No future GT informs prediction.
* `stale_summary.csv`: all-available and paired-history comparisons with explicit
  denominators; `survival_horizons.csv`: consecutive valid prefix before first
  failure, with right-censor flag. Later IoU recovery does not extend the prefix.
  These are oracle-initialized localization diagnostics, **not AP or tracking accuracy**.
  Unannotated targets are excluded, so false persistence after disappearance is
  outside this test. Long horizons have smaller, different cohorts.
* `camera_pairs.csv`, `camera_pair_summary.csv`: simultaneous pair distances and
  whether their complete bounding union fits one crop. `camera_centered.csv`:
  rounded and clamped target-centered crop, full/intersecting counts, class sets
  and next-frame full retention. Geometric opportunity does not imply legal
  movement from an arbitrary prior camera state.
* `gallery_index.csv`: every annotation's visual evidence page. All galleries have
  columns **source 1:1 / L0 nearest-neighbor x4 / L1 nearest-neighbor x2 / L2 1:1**.
  Every row shows the same object plus approximately 8 source pixels of context.
  L0/L1/L2 are extracted from actual official 960x540 rendered observations, with
  object-centered legal L1/L2 crops. Integer sampling grids cause a few pixels'
  padding difference. No smoothing/sharpening is used to enlarge coarse pixels.
  Source and L2 repeat intentionally to verify native information equivalence.
* `figures/comparison_*.png`: a boundary-free middle annotation per class when
  available. `galleries/*.png`: all 259 annotations at all three levels, five rows
  per page. `sequence_contact.png`: all 25 frames with GT labels (context, not
  fine-detail evidence). `visibility_timeline.png`, `trajectories.png`: temporal
  and geometric views. Inspect PNGs at native display size for sampling detail.

The script asserts contiguous paired files, source dimensions, schema, class validity,
positive in-frame geometry, unique class/frame rows, metadata/count agreement,
expected totals, and unchanged authoritative input hashes. Every L0 image is also
checked pixel-for-pixel against direct `INTER_AREA` resizing. All generated files
stay within this directory. A rerun is deterministic for unchanged inputs and package
versions; hashes capture evaluator and probe inputs as well as the full dataset.
