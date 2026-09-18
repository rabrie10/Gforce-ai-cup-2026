# Execution record

Run from repository root in PowerShell.

## Preflight

```powershell
git branch --show-current
git status
git log -5 --oneline
git show --stat --oneline --summary 6794da6
git show --stat --oneline --summary d1d195b
```

State: clean `drone-motion-architecture-experiments`, HEAD `6794da6`. Similarity
was already checkpointed, so no new similarity commit was needed. Existing packages
and validations were inspected read-only. No discrepancy from the documented state.

## Experiment

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_affine_oracle/run_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_affine_oracle/validate_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_affine_oracle/summarize_experiment.py
```

The runner was invoked several times while correcting non-scientific integration
errors: control string/int comparison, reused IoU helper name, loaded-control key
types, LOO CSV schema, and NumPy version metadata. Each failed before final result
completion; no estimator, threshold or interpretation changed in response to target
metrics. The final run passed controls and preservation checks. Existing local
Matplotlib dependencies required elevated filesystem access; no network/install ran.
A subsequent complete run compared SHA-256 hashes before/after for all 22
runner-generated result JSON, CSV and PNG artifacts; no hash changed.

## Final checks

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_affine_oracle/validate_experiment.py
git add -- drone-flyby/architecture_experiments/motion_affine_oracle
git diff --cached --check
git diff --cached --stat
git diff --name-only HEAD -- drone-flyby/architecture_experiments/motion_oracle drone-flyby/architecture_experiments/motion_similarity_oracle
git status --short
git branch --show-current
```

Only the affine package is staged. No commit, merge or push was run. Nine figures
were inspected with the local image viewer. Homography and image-derived GMC were
not started.
