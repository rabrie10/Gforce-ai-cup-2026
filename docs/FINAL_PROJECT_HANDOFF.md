# Final project handoff — Nordic AI Cup 2026, Drone Flyby

Prepared 2026-09-20 after the Final Evaluation.

## 1. Final submitted commit

| | |
|---|---|
| Repository | https://github.com/rabrie10/Gforce-ai-cup-2026 |
| Branch | `claude/hosted-l1l2-audit` (pushed) |
| Commit at submission time | `f3c247fb68323f6202a21d19b6f835e2c869a23d` |
| Head after handoff docs | `bbbd85b` (docs + restored serving file; no behaviour change) |
| Solution docs | `drone-flyby/SOLUTION_README.md` |

The organiser's root `README.md` is unchanged.

## 2. Official result

| | |
|---|---|
| Score | **0.024644923331075038** |
| Attempt UUID | `c594011b82984f9b9c0cf563ee34e1c9` |
| Type | FINAL EVALUATION (one attempt only, per competition rules) |
| Window | 2026-09-20T13:21:30Z → 13:22:54Z |
| Errors | 1 — transport timeout at frame 197 |
| Best validation | 0.03182123481701167 (`a23ca0fbf1124a3c852680c8a39d4d53`) |

Full record with provenance flags: `Nordic-AI-Cup-2026-Final-Archive/evaluations/OFFICIAL_RESULTS.json`.
Some earlier V6 scores exist only as session notes and are flagged `provenance=session_notes`;
their attempt UUIDs were never captured and must not be treated as authoritative.

## 3. Submitted architecture and runtime configuration

V6 pipeline with the V7 discovery detector and one changed gate:

```
V6_DETECTOR=/workspace/v7_e15_snapshot.pt
V6_CLASSIFY_MIN_PX=16      # V6 default was 22
V6_TARGET_MIN=0.5          # unchanged
V6_DET_CONF=0.15           # unchanged
V6_ENSEMBLE=0
V6_DIAGNOSTIC_CAPTURE=0
```

Served by `uvicorn v5.gpu.endpoint_v6:app --host 0.0.0.0 --port 9053 --workers 1`.

**The camera receive-clamp (`V6_RECV_CLAMP`) is NOT part of the submission.** It scored 0.0124
versus 0.0318 and was reverted before the Final Evaluation. The clamp variant is preserved as
`drone-flyby/architecture_experiments/v7_object_centric/NOT_SUBMITTED_pipeline_v6_cameraclamp.py`.

Serving file `drone-flyby/v5/gpu/pipeline_v6.py`:
* sha256 `6f9be14ef9b59875f2b1c6340b321ccb32b31c33489bc9c071a877c865bb83de` (CRLF, as stored on the pod)
* sha256 `984303a2153902e7466868fd425dd3d1c9d20bb893a01bf3a5353c3ce5d0e14c` (LF, as stored in Git)
* Verified byte-identical apart from line endings.

## 4. Local backup location

`C:\Nordic-AI-Cup-2026-Final-Archive`

```
checkpoints/    model weights + custom heads (hash-verified against Runpod)
training/       training run logs, metrics, args (training_runs.tgz)
datasets/       v7_objects.tgz (masks), dataset_v7.tgz (full 3600+420 generated dataset)
evaluations/    OFFICIAL_RESULTS.json
diagnostics/    ev_diagnostics.tgz, ev_audit.tgz, ev_incident.tgz
deployment/     SUBMITTED_pipeline_v6.py, scripts_cfg.tgz (deploy/rollback/env, secrets stripped)
experiments/    NOT_SUBMITTED_pipeline_v6_cameraclamp.py
logs/           logs.tgz (service + training logs)
source/         repo-all-refs.bundle (complete Git history, all refs)
documentation/  copies of the handoff docs
MANIFEST.json, SHA256SUMS.txt, README.md
```

Long-filename evidence (hosted captures) is stored as `.tgz` to avoid the Windows 260-character
path limit. Original→archived path mapping is in `MANIFEST.json`.

## 5. Checkpoints — all hash-verified against Runpod

| file | sha256 | role |
|---|---|---|
| `v7_e15_snapshot.pt` | `52b9fcef…a888b8ba` | **SUBMITTED detector** |
| `v7_final_e30.pt` | `cd370c1f…8514edd1` | epoch-30 alternative (not submitted) |
| `ms1_best.pt` | `e5172a37…a6888b8ba`* | V6 baseline detector |
| `v7a_last_e30.pt` | `1d2f2a15…da0d968a` | last epoch of the V7 run |
| `bg_head.npz` | `b1d0f4db…82e377910` | target/background head (custom) |
| `visual_head.npz` | `b1c04093…eff3b387d6` | 16-class visual head (custom) |
| `dinov2_224.onnx` | `ddacda63…63099b9ea` | DINOv2 encoder export |
| `discovery.onnx` | `66a77dfb…59c06dd1fb` | legacy discovery export |
| `bg_head_data.npz` | `ffff63c5…f4a9f90f6` | bg-head training features |

*exact value `e5172a375c6dcd8f72d84779fef27a48afba7aff0c9e5cec2422caa6a888b8ba`.
Full values: `checkpoints/LOCAL_VERIFIED_HASHES.json`. All 9 matched their Runpod source.

Foundation weights that can be re-downloaded: `yolo11s.pt` (Ultralytics COCO), `yolo11l-obb.pt`,
`yolov8l-worldv2.pt`. DINOv2 is preserved as the ONNX export above rather than re-exported.

## 6. Dataset and reproducibility

* Helsinki source (25 4K frames + annotations) — **tracked in Git**, spot-verified identical to
  the Runpod copy.
* Object masks (`assets/v7_objects`) — archived, irreplaceable (encodes the GrabCut vs
  soft-matte decisions and the manual QC rejections).
* Generated V7 dataset — **fully archived** (`datasets/dataset_v7.tgz`, 1.09 GB, hash-verified),
  3600 train / 420 val.
* Generator, seeds, split rules, training config and command — in Git.
* Bit-identical regeneration has **not** been verified; the dataset is archived rather than
  assumed reproducible.

## 7. Diagnostic evidence

Archived and hash-verified: the 20 genuine hosted L1/L2 captures with sidecars, the frozen manual
annotations (`manual_annotations_frozen.json`, sha256 `0dc88cb6…`), failure matrix, contact sheets,
overlays, V6/V7 comparisons, gate ablations, the transport-incident forensics and the retry/final
run telemetry. The frozen annotations, failure matrix and audit reports are also in Git; the raw
evaluator imagery is **not** committed (redistribution permission unverified).

## 8. Security

* No secrets found in tracked files, in Git history, or in the archive (patterns scanned; values
  never printed).
* No private keys are stored in the repo or archive. Deployment scripts document environment
  variable names only.
* **Action required: rotate the RunPod API key.** It was printed to a session transcript during
  the incident investigation on 2026-09-20 and must be considered exposed.

## 9. Post-submission obligations

The root `README.md` states: *"the top 5 highest-ranking teams will be asked to submit their
training code and the trained models for validation no later than September 20 at 20:00 CEST"*.

* No rule requiring the **endpoint to stay online** after Final Evaluation was found in the
  repository documentation. Absence of a rule in these files is not proof none exists elsewhere
  (portal terms, e-mail) — treat as unverified.
* The top-5 deliverable (training code + trained models) is fully satisfied by this archive plus
  the pushed branch, and does not require the pod.
