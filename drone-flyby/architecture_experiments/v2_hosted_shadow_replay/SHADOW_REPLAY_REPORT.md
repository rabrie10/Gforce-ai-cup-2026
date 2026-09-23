# Architecture V2 deterministic hosted-image shadow replay

## Executive result

**OUTCOME B.** The mandated primary recognition replay from proposal coordinates recorded to 0.1 pixel does not reproduce every hosted recognition value, so the strict classification is Outcome B. The disagreement is fully attributable to telemetry precision changing integer crop boundaries: the secondary end-to-end replay with unrounded detector coordinates reproduces hosted recognition exactly at recorded precision. This is not evidence of model/config/runtime drift, but the primary gate does not permit a domain-generalization conclusion.

The earliest observable disagreement is **recognition stage**.
The comparison is strictly internal replay reproducibility; there is no hosted ground truth.

## 1. Frozen artifact verification

All required hashes match: **True**.

| Artifact | Actual SHA256 | Result |
| --- | --- | --- |
| `drone_yolo11n_l0.pt` | `4c44e404e03673e8aa15fc85baf90f2d09b67d90c1fd94c9b6c9a4c1f35204ce` | PASS |
| `drone_yolo11n_l0.onnx` | `d480d3369feba50faa7d1ef0d901f6e5d96f564bf0953bd058f1de42a0ac2ca9` | PASS |
| `reference_gallery.npz` | `ef226de6a9bd624a1e7fd60291812677c5d7815f3a667bfea607b0158867b67e` | PASS |

No inference ran until this gate and the runtime manifest had been written.

## 2. Runtime and configuration manifest

- Branch at replay: `drone-v2-shadow-replay`
- HEAD at replay: `afbe0a6c47662b438362aafa01343224f85b11c9`
- Frozen V2 commit: `6fb764e5b9fb5f7eefcb832c72b35553165219e5`
- Python: `3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]`
- NumPy: `2.5.3`
- OpenCV: `5.0.0`
- ONNX Runtime: `1.30.0`
- Torch: `2.14.0+cpu`
- Torchvision: `0.29.0+cpu`
- Proposal backend: `OnnxYoloProposer` with providers `['CPUExecutionProvider']`

The complete effective proposal/recognition config, relevant environment/thread
variables, package versions, providers, and artifact sizes are in
`runtime_manifest.json`.

## 3. Image-byte verification

- Images processed: **50**
- Exact expected frame indices: `0, 5, ..., 245`
- All decoded as exactly 960x540: **True**
- SHA256 recorded for every image: **True**
- Source bytes modified or re-encoded: **False**

`image_manifest.csv` contains the per-file hashes, byte sizes, and dimensions.

## 4. Discovery/proposal replay

- Agreement at hosted telemetry precision: **True**
- Exact proposal-count match rate: **100.0%**
- Proposal-count MAE: **0.000000**
- Hosted/replay proposal totals: **41 / 41**
- Both zero: **24 frames**
- Hosted zero, replay nonzero: **0 frames**
- Replay zero, hosted nonzero: **0 frames**
- Ordering match rate: **100.0%**
- Matched IoU median/min: **0.993929945 / 0.985860526**
- Bbox max-absolute-error median/p95/max: **0.0415833 / 0.0495361 / 0.0499146 px**
- Score absolute-error median/p95/max: **2.09748e-05 / 4.69366e-05 / 4.96685e-05**

Ordinal matching is used when the geometric assignment preserves ordering;
otherwise the CSV labels deterministic global-greedy IoU matching explicitly.
No floating-point difference is hidden behind the agreement label.

## 5. Recognition replay

The primary comparison feeds the **original recorded hosted proposal boxes** into
the frozen local crop and recognition path. Locally replayed proposal boxes are
not used here.

- Agreement at hosted telemetry precision: **False**
- Hosted frames with proposals: **26**
- Crop totals hosted/replay: **41 / 41**
- Observation-count match rate: **100.0%**
- Background-rejection agreement, per proposal: **100.0%**
- Rejection totals hosted/replay: **3 / 3**
- Observation totals hosted/replay: **38 / 38**
- Top-class agreement rate where both accept: **94.7%**
- Top-posterior absolute-error median/p95/max: **3.41491e-05 / 0.00422249 / 0.070938**
- Quality absolute-error median/p95/max: **2.8343e-05 / 0.0164502 / 0.043733**

The primary mismatch is localized to telemetry precision:

- Integer crop bounds changed after reconstructing from 0.1-pixel boxes: **5/41 proposals**
- Primary disagreement frames: **[50, 80, 100, 210, 225]**
- Frames with changed integer crop bounds: **[50, 80, 100, 210, 225]**
- Every primary disagreement frame has a changed crop boundary: **True**
- Secondary full-precision detector-box sensitivity agrees: **True**
- Secondary top-class agreement: **100.0%**
- Secondary top-posterior absolute-error max: **4.99524e-05**
- Secondary quality absolute-error max: **4.89919e-05**

This secondary check is not substituted for the required primary comparison. It
shows that the primary mismatch is caused by crop reconstruction from serialized
coordinates rather than by the frozen detector, recognizer, image bytes, model
artifacts, or effective local config.

Per-crop hosted objectness and margin, rejected-crop hosted class/posterior/quality,
and non-top hosted posterior values are **NOT OBSERVABLE**. Per-frame mean
objectness and mean margin are compared because telemetry records them.

## 6. Earliest disagreement and classification

- Earliest stage: **recognition stage**
- Final classification: **OUTCOME B**

Ranked causes of the observed primary disagreement:

1. **Preprocessing mismatch from telemetry-rounded proposal geometry — CONFIRMED.**
2. **Model artifact or effective config mismatch — CONTRADICTED** by hashes and the exact full-precision sensitivity replay.
3. **Image decode difference — CONTRADICTED** by exact discovery and full-precision recognition reproduction.
4. **ONNX/runtime/library/thread/provider behavior — NOT SUPPORTED.**
5. **Other deployment/runtime drift — NOT SUPPORTED on these samples.**

## 7. Confidence and limitations

Confidence is **high for these 50 sampled images at recorded telemetry precision**.
Hosted boxes are rounded to 0.1 pixel and scalar inference values to four decimals.
Recognition crops therefore use the exact recorded boxes, but those records are
not the unavailable full-precision in-memory proposal boxes. The captured set is
a stride-5 sample, not all endpoint inputs.

There is no hosted ground truth. Hosted proposal recall, detection accuracy,
classification accuracy, IoU to hidden objects, hidden class-specific AP, and
hidden object scale remain **NOT OBSERVABLE**.

## 8. Exactly one next action

**Extend diagnostic telemetry to preserve full-precision proposal crop bounds and a SHA256 for each 64x64 recognition crop before any future authorized hosted capture.**

This action is recommended only; it was not implemented.

## Reproduction

From the repository root:

```powershell
& 'drone-flyby/.venv/Scripts/python.exe' 'drone-flyby/architecture_experiments/v2_hosted_shadow_replay/run_shadow_replay.py'
```

The command fails closed on artifact hash drift, image-count/dimension errors,
missing sampled telemetry, or internally inconsistent output row counts.
