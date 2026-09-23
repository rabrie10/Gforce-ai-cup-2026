# Commands and execution record

Working directory:
`C:\Users\mbnas\.vscode\nordicAI\Gforce-ai-cup-2026`

## Preflight

```powershell
git status --short
git branch --show-current
git log -3 --oneline
rg --files -g AGENTS.md -g '!**/.cache/**' -g '!**/.deps/**' -g '!**/node_modules/**'
git ls-files drone-flyby/architecture_experiments/motion_oracle
```

The working tree was clean on `drone-motion-architecture-experiments`, HEAD
`d1d195b`. No branch switch was needed. No applicable AGENTS.md was found.
Read-only `Get-Content` and Python inspection commands examined the reviewed
runner, validator, lockfile and generated tables. No prior artifact was edited.

## Scientific commands actually run, in order

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_similarity_oracle/run_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_similarity_oracle/validate_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_similarity_oracle/summarize_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_similarity_oracle/run_experiment.py --plots-only
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_similarity_oracle/validate_experiment.py --supplementary-only
```

The full runner passed all control checks, wrote predictions and generated plots.
The independent validator passed. Supplementary spatial slopes and the numerical
appendix were then added; their separate validation passed without repeating the
already passed fit/prediction checks. The plot-only command fixed an overlapping
colorbar using saved results; no prediction was refit or retuned. Nine final PNGs
were visually inspected using the local image viewer.

The runner and plot-only command used elevated tool permissions solely to read
the existing Matplotlib dependencies. No package installation, web research,
image-based motion estimation or detector execution was performed. The final
validator source runs both validation phases; for fresh reproduction use the
README order: runner, summarizer, validator.

## Artifact checks

```powershell
git add -- drone-flyby/architecture_experiments/motion_similarity_oracle
git diff --cached --check
git diff --cached --stat
git status --short
git diff --name-only HEAD -- drone-flyby/architecture_experiments/motion_oracle
git branch --show-current
```

Only the new package is staged. The reviewed package has no Git diff, and all
60 preserved-input SHA-256 hashes match. Required reports, JSONs, CSVs, scripts
and figures are staged; dependency/font/bytecode caches are excluded. No commit,
push or merge was run. The affine experiment and image-derived GMC have not begun.
