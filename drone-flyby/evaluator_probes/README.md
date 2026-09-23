# Evaluator probes

These are controlled evaluator experiments, with no detector or solution code.
Run commands from the repository root. Python 3.12 was used for the recorded run.

## Setup

Windows PowerShell (with a working Python launcher):

```powershell
py -3.12 -m venv drone-flyby/.venv
& drone-flyby/.venv/Scripts/python.exe -m pip install -r drone-flyby/evaluator_probes/requirements.lock.txt
& drone-flyby/.venv/Scripts/python.exe drone-flyby/local_evaluator.py --oracle
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/run_all.py
```

Linux/macOS:

```sh
python3.12 -m venv drone-flyby/.venv
drone-flyby/.venv/bin/python -m pip install -r drone-flyby/evaluator_probes/requirements.lock.txt
drone-flyby/.venv/bin/python drone-flyby/evaluator_probes/run_all.py
```

The original `drone-flyby/requirements.txt` also suffices, but its version ranges
can install a different scorer. The lock file records this experiment's exact
dependencies; it does not assert which version the hosted evaluator runs.

## Operation

The complete runner first executes the supplied oracle CLI and checks all class
scores. It stops if the oracle is not 1.0. It then runs A-G, H-K, and the justified
absent-class check. It starts and stops its own HTTP server bound to `127.0.0.1`
on ephemeral ports; there is no need to run `api.py`. Allow local loopback traffic.
Expect roughly 4-7 minutes on a desktop, depending on PNG decoding/encoding and
CPU scheduling. Timing results are measurements and will vary across runs.

`probe_results.json` contains the measurements, including per-class AP, all
request camera/feedback states, counters, each RTT, exact elapsed-time recurrence
observations, evaluator diagnostics, dependency versions, and authoritative
source/data hashes. Image payloads are omitted. `PROBE_RESULTS.md` is the reviewed
human-readable report for the recorded run. Running the suite replaces JSON;
regenerate the report with `python drone-flyby/evaluator_probes/report.py` after
reviewing new measurements. The report generator is also called by the full runner.
Use `--output PATH` to keep a separate JSON run; its report uses the same basename
with `.md`. The default output uses `PROBE_RESULTS.md`.

Temporary synthetic scenes use actual 3840x2160 PNGs and annotation JSON in the
OS temporary directory. They are removed on normal completion. The official
Helsinki dataset and evaluator are never patched, copied over, or edited.
Existing ignore rules already exclude `.venv`, bytecode, and caches.

## Measurement boundaries

Pure scoring experiments import `local_evaluator.score`. Protocol and timing
experiments use the actual `replay`, `requests`, response DTO, image renderer,
camera, and scorer. Synthetic scenes enter through the supplied loader's absolute
scene-path support. Timing uses all 25 official Helsinki frames, actual sleeps in
the HTTP handler, and real monotonic time; no mocked clock or simulated latency.

A read-only Python trace observes the replay locals immediately after its next
index calculation. It does not replace any function or variable. The small tracing
overhead, host scheduling, image preparation, and HTTP serialization are included
in elapsed time. RTT begins after image preparation; it is not total per-frame
processing time. The reported median is the statistical median; the supplied CLI
uses the upper middle observation for an even number of requests.

NaN cannot be emitted as a numeric value in strict JSON. The suite tests numeric
NaN/Infinity directly against the official DTO and records strict encoder refusal;
it separately sends legal JSON strings and an overflowing JSON exponent (`1e999`)
over HTTP. It never emits bare nonstandard `NaN`/`Infinity` JSON tokens.

## Score-only version parity

From repository root, using the existing environment and recorded results:

```powershell
& drone-flyby/.venv/Scripts/python.exe -m pip install --no-deps --target drone-flyby/evaluator_probes/.cache/scorer-1.7.2 faster-coco-eval==1.7.2
& drone-flyby/.venv/Scripts/python.exe -m pip install --no-deps --target drone-flyby/evaluator_probes/.cache/scorer-1.8.0 faster-coco-eval==1.8.0
& drone-flyby/.venv/Scripts/python.exe drone-flyby/evaluator_probes/scorer_parity.py
```

Install commands are for first setup; existing correctly installed target folders
can be reused. The runner verifies exact imported version and module path in a
fresh process for each version. Other dependencies and the existing `.venv` stay
unchanged. It runs only Oracle/A/B/C/D/E/G/L scoring cases, using the same inputs
as the full suite. A rechecks scoring on identical global predictions with crop
geometry recorded as metadata; it does not replay the earlier camera experiment.
No HTTP endpoint, camera operation, protocol test, or latency sweep is invoked.

Expect seconds rather than minutes. The runner adds `scorer_version_parity` to
`probe_results.json`, preserves existing measurements, and regenerates the report
with the parity section. It compares unrounded numeric results exactly and lists
every difference. Isolated packages and per-version intermediate JSON remain in
the already-ignored `.cache/` directory. A subsequent full-suite run replaces the
main JSON; run the parity command afterward to attach a fresh comparison.
