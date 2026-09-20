# Baseline 1 failure audit

This package is a read-only forensic reconstruction of the frozen Baseline 1
result at commit `129564f855060ea373dc8421701f97631162a828`.

Run from the repository root with the bundled Python runtime (or any Python
3.12 environment containing NumPy and Pillow):

```powershell
& 'C:\Users\mbnas\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  drone-flyby/architecture_experiments/baseline1_failure_audit/run_audit.py
```

The script consumes saved predictions and existing evidence only. It does not
load the checkpoint, retrain, call an endpoint, tune a threshold, or alter any
prior artifact. See `BASELINE1_FAILURE_AUDIT.md` for conclusions and
`audit_results.json` for the machine-readable result.

