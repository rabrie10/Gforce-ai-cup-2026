# Motion Oracle Experiment — Helsinki only

Read [MOTION_ORACLE_EXPERIMENT.md](MOTION_ORACLE_EXPERIMENT.md) for the decision and
[motion_oracle_results.json](motion_oracle_results.json) for machine-readable results.
This package reads annotation JSON and prior audit artifacts only. It does not read
images, import the evaluator, run inference, train, or implement a tracker.

## Reproduce from the repository root (PowerShell)

```powershell
& drone-flyby/.venv/Scripts/python.exe -m pip install --target drone-flyby/architecture_experiments/motion_oracle/.deps -r drone-flyby/architecture_experiments/motion_oracle/requirements.lock.txt
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/run_experiment.py --baseline-only
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/run_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/summarize_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/validate_experiment.py
```

The existing Python is 3.12.14. Installation is needed only for missing local
dependencies. The experiment itself has no network calls. The baseline-only gate
already passed; it need not be repeated when reviewing the supplied results.
The full runner always checks it before new predictions. On this host, sandbox
access to pip-installed files required running the plotting command with elevated
tool permissions. This is an environment access issue, not an experiment feature.

`validate_experiment.py` reconstructs all saved predictions with scalar arithmetic
and a separate center/half-width IoU implementation. It also perturbs future target
GT at all 243 eligible observation origins, recomputes donor estimates, and checks
that the feature API gives unchanged predictions. That part can take several
minutes. To validate only supplementary tables after the full validator has passed:

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/validate_experiment.py --derived-only
```

## Definitions and files

- `prediction_results.csv`: all 2,456 GT origin/future pairs for each of five models
  (12,280 rows), including unavailable cases. Source boxes use continuous `xyxy`
  dimensions without `+1`. Predictions clip once to `[0,3840] x [0,2160]` at output;
  invalid/collapsed geometry gets IoU zero and remains an available failure.
- `horizon_summary.csv`: all-origin eligibility, model availability, IoU success
  count/fraction, mean/median IoU, center/size errors, for every supported horizon.
  `valid` always means IoU >=0.50. `valid_fraction` divides by available count;
  availability divides by all-origin eligible count. Blank metrics mean unavailable.
- `common_cohort_summary.csv`: identical class/origin/horizon intersections across
  all five models; all listed eligible cases are available. It covers h=1 through
  h=23 and contains 2,213 keys per model (11,065 prediction rows). At h=24 only
  single-observation methods can be available, so there is no five-model cohort.
- `pairwise_common_cohorts.csv`: every pair's own available intersection, including
  extra origins usable without target history. Also records discordant successes.
- `per_class_summary.csv`, `size_summary.csv`, `target_size_summary.csv`,
  `boundary_summary.csv`: five-model common cohorts, stratified by horizon.
  Mine roller has no two-observation prediction origin with future GT and hence no
  common-cohort row, but is present in the full transition and prediction tables.
- Size bins match the audit: source short side divided by four (L0 apparent size),
  `<4`, `[4,8)`, `[8,16)`, `[16,32]`, `>32`. Primary stratification uses origin size;
  future-size stratification is evaluation only. `any_boundary` means a box within
  one pixel of an edge at t-1, t, or the evaluated target frame (where present).
  This is possible truncation, not proof of clipping or an occlusion label.
- `center_error`: un-clipped latent predicted center against evaluated GT center,
  in source pixels. `clipped_center_error` is also saved. Normalized error divides
  by evaluated GT short side. Width/height errors are signed latent predicted size
  minus evaluated GT size; summary columns report absolute means. Future target
  size is never a feature. Clipping is not fed back into the sequential state.
- `donor_updates.csv`: 408 estimates (24 transitions x 16 excluded classes plus
  24 all-object descriptive estimates), with exact donor lists, dx/dy medians,
  coordinate MAD (unscaled), median Euclidean donor residual, size ratios and
  minimum-one-donor support flags. Actual predictions use only excluded-class
  estimates. `ALL` is strictly descriptive. Missing transition 0 has no estimate.
- Each prediction records `target_feature_frames` and `update_ids`. M2 uses only
  the origin transition; M3 uses future transitions; M4 uses both. M1/M4 require
  consecutive target history. No donor fallback exists; insufficient support is
  explicitly unavailable. Future other-object GT in M3/M4 is **ORACLE REFRESHED
  SHARED MOTION**, not a deployable measurement.
- `motion_decomposition.csv`: all 243 object transitions and all-object descriptive
  residuals. `shared_motion_by_frame.csv`: 24 shared vectors, direction in image
  coordinates (positive y downward), dispersion and scale diagnostics.
- `loo_spatial_generalization.csv`: 243 independently held-out one-step targets,
  including those without previous target velocity. This corresponds to M3's
  one-step all-origin cohort, not its smaller five-model cohort.
- `additional_diagnostics.json`, `scale_by_class.csv`: used-donor dispersion,
  descriptive spatial correlations and boundary-filtered size behavior.
- `baseline_reproduction.json`, `validation_results.json`,
  `derived_validation_results.json`: reproduction and independent check results.
- `figures/`: eight PNGs, all required plots plus scale behavior. The success curve
  is per-horizon IoU success, not a first-failure survival estimator; its changing
  cohorts are specified in the summaries.

The scientific report is written from these measurements; it is not automatically
rewritten by the runner. Input hashes cover all 25 annotation files, metadata and
four prior audit artifacts. Official images were neither accessed nor modified.

The local `.gitignore` excludes only `.deps/`, `.mplconfig/`, and `__pycache__/`.
Reports, JSON, CSV, source scripts, lockfile and PNG figures are review artifacts.
No source data copies or model weights are included. See [COMMANDS.md](COMMANDS.md)
for the execution record and environment-only failed attempts.
