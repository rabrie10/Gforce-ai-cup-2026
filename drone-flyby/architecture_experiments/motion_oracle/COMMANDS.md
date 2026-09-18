# Execution record

All commands below were run from
`C:\Users\mbnas\.vscode\nordicAI\Gforce-ai-cup-2026` in PowerShell.
Read-only source inspection additionally used `Get-Content`, `rg`, and Python
one-liners to inspect the existing audit, annotations and generated results.

## Repository preflight and branch

```powershell
git status --short
git branch --show-current
git log -5 --oneline
git branch --list drone-motion-architecture-experiments
git switch -c drone-motion-architecture-experiments
```

The working tree was clean. Initial branch: `drone-targeted-solution-research`.
Recent commits: `bfed48e`, `6129b04`, `84c30b1`, `b531337`, `401dc7b`.
The branch did not exist. The first switch attempt could not write `.git`; the
same command succeeded with elevated tool permissions. No merge was run.
Searches found no applicable `AGENTS.md`. An initial broad filesystem search hit
access-denied folders in unrelated evaluator caches; later searches excluded them.

## Dependency setup

```powershell
& drone-flyby/.venv/Scripts/python.exe -m pip --version
& drone-flyby/.venv/Scripts/python.exe -m pip install --target drone-flyby/architecture_experiments/motion_oracle/.deps matplotlib --no-deps
& drone-flyby/.venv/Scripts/python.exe -m pip install --target drone-flyby/architecture_experiments/motion_oracle/.deps 'matplotlib==3.10.8' --retries 0
```

The first installation failed because sandbox network access was unavailable.
The pinned installation succeeded with elevated permissions and installed the
versions preserved in `requirements.lock.txt`, entirely in experiment-local
`.deps/`. The existing project environment was not modified. Neither the project
environment nor bundled runtime initially had Matplotlib; the system `python`
command was unavailable and `py` pointed at a missing Python installation.
These failed environment probes did not affect any scientific result.

## Scientific execution, in order

```powershell
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/run_experiment.py --baseline-only
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/run_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/run_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/validate_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/summarize_experiment.py
& drone-flyby/.venv/Scripts/python.exe drone-flyby/architecture_experiments/motion_oracle/validate_experiment.py --derived-only
```

Baseline-only passed exactly, including all 2,213 audit rows. The first full run
wrote prediction tables but could not access pip-installed Matplotlib within the
sandbox. The second identical full-run command, with elevated tool permissions,
successfully generated all eight plots and checked unchanged input hashes.
The full independent validator passed; it ran while supplementary analysis was
being prepared. After adding the supplementary tables/checks, `--derived-only`
validated those without repeating the already passed prediction/leakage checks.
The final validator source includes both validation phases for fresh reproduction.

For a fresh run, use the cleaner reproduction order in README: runner, summarizer,
validator. The measurements and baseline were not restarted after the user's
continuation request. All eight PNGs were inspected using the local image viewer.

## Review artifact checks

```powershell
git check-ignore drone-flyby/architecture_experiments/motion_oracle/motion_oracle_results.json drone-flyby/architecture_experiments/motion_oracle/prediction_results.csv drone-flyby/architecture_experiments/motion_oracle/figures/iou_success.png
git add -- drone-flyby/architecture_experiments/motion_oracle
git diff --cached --check
git diff --cached --stat
git status --short
git branch --show-current
```

`check-ignore` produced no matches. The local ignore list contains only `.deps/`,
`.mplconfig/`, and `__pycache__/`. Required scientific outputs are staged for review,
with no changes outside the experiment directory. No commit, push or merge was run.
