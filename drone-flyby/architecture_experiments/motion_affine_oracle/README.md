# GT-only strict leave-one-object-out affine motion

Read [MOTION_AFFINE_ORACLE_EXPERIMENT.md](MOTION_AFFINE_ORACLE_EXPERIMENT.md) for
the decision evidence and [MEASUREMENTS.md](MEASUREMENTS.md) for complete numerical
tables. This is a Helsinki-only oracle geometry experiment. Future GT of other
objects refreshes T/S/A shared motion; future target GT is evaluation-only.

## Reproduce from the repository root

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_affine_oracle/run_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_affine_oracle/summarize_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_affine_oracle/validate_experiment.py
```

The runner reads the reviewed packages and source annotations without writing to
them. It hashes all 93 reviewed/source files before and after execution. Python
bytecode is disabled and Matplotlib uses this package's `.mplconfig/`. Existing
locked plotting dependencies are read from the reviewed packages; neither prior
experiment script is executed as a program. No network, image read, evaluator,
model inference or training occurs.

The seven models are loaded controls `CV`, `T_FIXED`, `T_RES_CV`, `S_FIXED`,
`S_RES_CV`, plus `A_FIXED` and `A_RES_CV`. A_FIXED transports centers with future
held-out affine transforms and fixes origin dimensions. A_RES_CV uses exactly the
similarity experiment's frozen image-axis residual convention and target size CV.
Every T/S/A result is **ORACLE REFRESHED SHARED MOTION**.

Affine uses other-object centers only: `p' = A p + t`, full 2x2 A, Huber IRLS
delta 3 source pixels, 20 fixed updates, weighted centering and weighted least
squares. Initial/final spatial support uses >=3 donors, RMS spread >=100 px,
minor spread >=25 px, covariance eigenvalue ratio >=0.001 and condition <=10;
final effective N must be >=3. Transform validity requires finite parameters,
positive determinant, singular values in [0.5,2] and map condition <=4. These broad
limits were frozen before target evaluation. Failures are unavailable without
fallback. Helsinki has no failed fit; synthetic fixtures exercise each class.

`prediction_results.csv` contains 17,192 rows (2,456 origin/future pairs x seven
models), including unavailable rows. `common_cohort_summary.csv` contains 2,213
identical keys per model, h=1..23. `affine_transforms.csv` stores 384 fits with all
donor IDs, final weights/residuals, support, A/t, determinant, singular values,
anisotropy, condition and polar/SVD diagnostics. `loo_spatial_generalization.csv`
contains 243 targets each for translation, similarity and affine. Stratified and
pairwise CSVs retain full auditability. Continuous xyxy, clipping and IoU conventions
are unchanged from the reviewed packages.

The independent validator reconstructs affine with scalar covariance algebra,
reconstructs every prediction and IoU, validates every summary/cohort, confirms
all controls, runs deletion/poisoning/restricted-history tests, repeats every class
fit deterministically, checks synthetic transform/support/reflection/stability
fixtures, and rechecks all preservation hashes.

The local `.gitignore` excludes only `.deps/`, `.mplconfig/`, and `__pycache__/`.
Reports, scripts, JSON, CSV and nine PNG figures are review artifacts.
