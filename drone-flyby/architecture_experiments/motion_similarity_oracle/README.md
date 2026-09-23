# GT-only strict leave-one-object-out similarity motion

Read [MOTION_SIMILARITY_ORACLE_EXPERIMENT.md](MOTION_SIMILARITY_ORACLE_EXPERIMENT.md)
for interpretation and [MEASUREMENTS.md](MEASUREMENTS.md) for complete numerical
tables. All shared models use **ORACLE REFRESHED SHARED MOTION**: future GT of other
objects supplies each new transform. Future target GT is evaluation-only.

## Reproduction from the repository root

Use the existing Python 3.12 project environment and the plotting dependencies
already installed for the reviewed translation experiment:

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_similarity_oracle/run_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_similarity_oracle/summarize_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_similarity_oracle/validate_experiment.py
```

If dependencies are absent, install the reviewed lockfile into **this new package**:

```powershell
& drone-flyby/.venv/Scripts/python.exe -m pip install --target drone-flyby/architecture_experiments/motion_similarity_oracle/.deps -r drone-flyby/architecture_experiments/motion_oracle/requirements.lock.txt
```

No dependency installation was needed for this run. On this Windows host, access
to the existing plotting dependencies requires elevated tool permissions. That
changes filesystem access, not the scientific procedure. No script contains network
requests, image reads, evaluator imports, detector execution or training.

`run_experiment.py` hashes the 34 tracked reviewed-experiment artifacts and 26 source
annotation/metadata files, builds donor-only transforms, checks all three controls
against saved prior rows, then writes the new results and plots. The old scripts
are never executed or imported; the shared plotting dependency directory is read
only. Python bytecode writes are disabled, and Matplotlib uses a new local cache.

To redraw figures from saved results or validate supplementary slopes alone:

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_similarity_oracle/run_experiment.py --plots-only
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_similarity_oracle/validate_experiment.py --supplementary-only
```

## Models and conventions

| ID | Center prediction | Size prediction |
| --- | --- | --- |
| CV | independent target constant velocity | target width/height CV |
| T_FIXED | refreshed donor median translation | fixed origin size |
| T_RES_CV | refreshed translation + frozen target residual | target size CV |
| S_FIXED | refreshed donor similarity | fixed origin size |
| S_RES_CV | refreshed similarity + frozen target residual | target size CV |
| S_SCALE | same center as S_FIXED | origin size x product of future scales |
| S_RES_SCALE | same center as S_RES_CV | origin size x product of future scales |

`F_k(p) = [[a,-b],[b,a]]p + [tx,ty]`; scale is sqrt(a²+b²), rotation is atan2(b,a).
No reflection, anisotropic scaling, shear, affine or projective model is fitted.
Similarity residual is `r = c_t - F_t(c_(t-1))`, measured from the final observed
transition and added unchanged in image-axis pixels after every future transform.
It is not rotated/rescaled as a separate state. Translation residual uses the same
convention and reduces to the reviewed target-velocity-minus-shared-velocity form.

The robust fit uses Huber IRLS with delta=3 source pixels and 20 fixed reweighting
steps followed by a final weighted solve. All donors remain recorded; none is
silently removed. Support requires >=3 donors, RMS spatial spread >=100 pixels,
minor-axis spread >=25 pixels, and covariance eigenvalue ratio >=0.001. These
checks are repeated on final weighted support with effective N >=3. They are
conservative coverage conditions, not the mathematical two-point minimum for
identifying a similarity. Parameters were fixed before target evaluation.

Size ablations multiply dimensions only by uniform scale. They do not rotate the
box and enlarge its enclosing AABB. S_RES_SCALE replaces, rather than adds to,
target size CV. This makes the size-policy comparison explicit.

As in the reviewed package: continuous xyxy geometry without +1; source frame
3840x2160; final-output clipping; invalid/collapsed geometry scores zero; latent
state is never recursively clipped. Center error uses latent predicted center;
clipped-center error is also saved. Normalization divides by evaluated GT short
side. Width/height errors are signed latent-size errors; summaries use absolute
means. Frame counts convert to time at 3 FPS.

## Review artifacts

- `prediction_results.csv`: 2,456 GT origin/future pairs x seven models = 17,192
  rows, including unavailable cases; exact feature times and donor-update IDs.
- `donor_transforms.csv`: 384 held-out class/transition fits, all donor identities,
  counts, support statistics, coefficients, scale/rotation, final weights and
  per-donor residuals. Missing/degenerate fits carry a reason and no coefficients.
- `horizon_summary.csv`: model-specific availability. `valid` means IoU >=0.50;
  valid fraction divides by available count; availability divides by eligible count.
- `common_cohort_summary.csv`: all-seven-model intersection, 2,213 keys per model,
  h=1..23. All its eligible rows are available. All-origin tables reach h=24.
- `pairwise_common_cohorts.csv`: each pair's own available intersection, including
  single-observation origins for pure shared models; records discordant successes.
- `per_class_summary.csv`, `size_summary.csv`, `target_size_summary.csv`,
  `boundary_summary.csv`: all-seven common cohorts stratified by horizon. Size bins
  are the audit's L0 short side (source/4): <4, [4,8), [8,16), [16,32], >32.
  Boundary is <=1 pixel from any image edge at t-1, t or evaluation frame. It marks
  possible truncation, not an occlusion label. Future size/boundary is never a feature.
- `loo_spatial_generalization.csv`: all 243 consecutive targets x T_FIXED,
  S_FIXED and S_SCALE, including single-observation origins. Spatial correlations
  are also reported after removing within-transition means to check temporal mixing.
- `spatial_slopes.csv`: univariate descriptive slopes, not a fitted affine transform.
- `similarity_oracle_results.json`, `control_reproduction.json`,
  `preserved_inputs.json`, `validation_results.json`,
  `supplementary_validation_results.json`: results, provenance and checks.
- `figures/`: nine PNGs covering matched-size success/IoU/center error, scale-size
  ablation, spatial vectors/correlations, held-out error distribution, transform
  parameters over time and h=6 per-class performance.

The independent validator uses scalar analytic weighted similarity reconstruction,
separate center/half-width IoU, every saved prediction, all summary/cohort rows,
target deletion/poisoning, restricted feature histories and synthetic known-transform
and degenerate-support fixtures. It does not reuse the primary numerical fit.

`MEASUREMENTS.md` is regenerated from saved CSV/JSON. The scientific report is
manually interpreted and must be reviewed if measurements change. The local ignore
file excludes only `.deps/`, `.mplconfig/`, and `__pycache__/`; all scientific outputs
are intended for Git review. No prior artifact is modified by these commands.
