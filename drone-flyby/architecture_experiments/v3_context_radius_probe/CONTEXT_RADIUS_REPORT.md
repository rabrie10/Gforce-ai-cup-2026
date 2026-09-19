# Frozen V2 local-context radius ablation

## Executive result

**CASE L — LOCAL CONTEXT MEMORIZATION SUPPORTED.** Activation recovery increased materially with preserved context and occurred across multiple classes in both replacement families.

This is the final forensic diagnostic before V3. It measures frozen V2 activation recovery as progressively larger native Helsinki rectangles are preserved around the same target, while all outside pixels are replaced.

## Experiment identity

- Branch: `drone-v3-context-radius-probe`
- Experiment input commit: `8a79714031f678ff8b5297616d3d8a1bb783b0cb`
- Frozen V2 lineage: `6fb764e5b9fb5f7eefcb832c72b35553165219e5`
- Valid native controls: **6/8**
- Random seed: `20260919`
- Context composites: **96**
- Runtime: OpenCV `5.0.0`, NumPy `2.5.3`, ONNX Runtime `1.30.0` on `CPUExecutionProvider`

## Native control check

| Class | Frame | Rank .30 | Rank .50 | Score .30 | Max IoU | >=0.01 | Proposals | Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| condor | 3 | 5 | 5 | 0.049266 | 0.869 | True | 2290 | True |
| helicopter | 8 | 1 | 1 | 0.936126 | 0.845 | True | 2128 | True |
| jammer | 1 | 19 | 19 | 0.005902 | 0.715 | False | 2379 | True |
| large_launcher | 9 | 1 | 1 | 0.976145 | 0.828 | True | 2112 | True |
| small_launcher | 12 | none | none | 0.000000 | 0.227 | False | 2013 | False |
| spacecraft | 10 | 3 | 3 | 0.877338 | 0.901 | True | 2091 | True |
| ta-ta | 17 | none | none | 0.000000 | 0.224 | False | 1847 | False |
| tank | 13 | 7 | 7 | 0.014326 | 0.936 | True | 2068 | True |

A valid recovery control was pre-defined as native-original rank <=20 at IoU >=0.30. Other targets remain reported but do not count in normalized recovery rates.

## Recovery curve normalized to native

| Family | Context | R@20 .30 | R@20 .50 | Activation recovered | Median rank delta | Median score ratio | Median IoU ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| helsinki | 1x | 37.5% | 25.0% | 16.7% | 15.5 | 0.23079585342746578 | 0.7302602037811021 |
| helsinki | 1.25x | 37.5% | 25.0% | 33.3% | 21.5 | 0.23663159755886407 | 0.8189401198321363 |
| helsinki | 1.5x | 50.0% | 50.0% | 33.3% | 5.5 | 0.20733510068018168 | 0.8375977888706971 |
| helsinki | 2x | 50.0% | 50.0% | 33.3% | 4.5 | 0.14712639248764833 | 0.9221164447464418 |
| helsinki | 4x | 62.5% | 62.5% | 83.3% | 2.0 | 0.8272595747641386 | 0.9633344047961669 |
| helsinki | 8x | 62.5% | 62.5% | 83.3% | 1.5 | 0.9718191429712405 | 0.966307428483314 |
| hosted_validation | 1x | 12.5% | 12.5% | 0.0% | 220.0 | 0.0010019659572709661 | 0.7146397492432077 |
| hosted_validation | 1.25x | 25.0% | 0.0% | 0.0% | 134.5 | 0.0017415465472427102 | 0.5963812040027578 |
| hosted_validation | 1.5x | 37.5% | 25.0% | 0.0% | 119.0 | 0.0030888499849684708 | 0.5953132163903072 |
| hosted_validation | 2x | 50.0% | 37.5% | 0.0% | 6.5 | 0.015287777915396742 | 0.6615359610356644 |
| hosted_validation | 4x | 62.5% | 62.5% | 16.7% | -1.0 | 0.15720680746977847 | 0.8339352231363547 |
| hosted_validation | 8x | 62.5% | 62.5% | 33.3% | -1.0 | 0.3848207090991503 | 0.9593804464756805 |
| all | 1x | 25.0% | 18.8% | 8.3% | 82.5 | 0.0066804673895554355 | 0.7146397492432077 |
| all | 1.25x | 31.2% | 12.5% | 16.7% | 30.0 | 0.010846998217095156 | 0.7180409111430134 |
| all | 1.5x | 43.8% | 37.5% | 16.7% | 6.5 | 0.021983467294562596 | 0.7370238505538547 |
| all | 2x | 50.0% | 43.8% | 16.7% | 4.5 | 0.02818114866061129 | 0.8433610887054532 |
| all | 4x | 62.5% | 62.5% | 50.0% | 0.0 | 0.4738044337698462 | 0.9259033402899991 |
| all | 8x | 62.5% | 62.5% | 58.3% | 0.0 | 0.7457684246597791 | 0.966307428483314 |

Primary recovery criterion (fixed before inference): top-20 at IoU >=0.30 and matching score >=50% of that target's native-original score.

## Result by target class and replacement family

| Class | Native valid | Helsinki minimum recovery | Hosted minimum recovery | Recovered in both |
| --- | --- | ---: | ---: | --- |
| condor | True | 1.5 | none | False |
| helicopter | True | 1.0 | 8.0 | True |
| jammer | True | none | none | False |
| large_launcher | True | 4.0 | 8.0 | True |
| small_launcher | False | none | none | False |
| spacecraft | True | 1.25 | none | False |
| ta-ta | False | none | none | False |
| tank | True | 4.0 | 4.0 | True |

## Rendering and controls

- Each target remains at its native source-frame coordinates and scale.
- The preserved rectangle is centered on the annotation bbox. For multiplier `m`, requested width/height are `m*bbox_width` and `m*bbox_height`; edges use floor/ceil and clip to source bounds. Requested and effective dimensions are recorded per row.
- Every pixel inside the effective rectangle is copied byte-for-byte from the native frame; everything outside comes from one deterministic replacement image.
- Hosted 960x540 captures are expanded to 3840x2160 with nearest-neighbor replication. Every composite then uses `cv2.INTER_AREA` for the exact 3840x2160 -> 960x540 L0 render.
- Frozen ONNX decoding uses class-score max pooling, confidence floor 1e-4, V2 IoU 0.55 NMS, and no production top-K budget.

## Interpretation gate

The gate was fixed in code before inference: CASE W if fewer than 4 native targets are top-20 at IoU >=0.30. Otherwise CASE L requires at least 3 target classes to recover in both background families, a combined 8x-minus-1x recovery-rate gain of at least 25%, and a higher median score-recovery ratio at 8x. All other valid-control outcomes are CASE N.

Observed combined activation recovery changed from **8.3%** at 1x to **58.3%** at 8x. Classes recovering in both families: **helicopter, large_launcher, tank**.

**Final classification: CASE L — local context memorization supported.**

## Limitations

- Rectangular replacement boundaries are an intervention artifact; increasing radius moves that boundary away but does not reproduce an unmodified whole frame until the rectangle reaches frame bounds.
- One physical Helsinki instance per class tests context dependence for those selected instances, not generalization to unseen instances.
- Hosted captures have no ground truth and are used only as replacement scenery; unrelated proposals can affect rank.
- Only one deterministic replacement image per family is used for each target to keep the final probe compact.

## Exactly one recommended next action

**Begin V3 discovery implementation.**

## Reproduction

```powershell
& 'drone-flyby/.venv/Scripts/python.exe' 'drone-flyby/architecture_experiments/v3_context_radius_probe/run_context_radius_probe.py'
```

The harness fails closed on branch mismatch, frozen-artifact hash drift, missing inputs, changed target selection, invalid preserved pixels, or unexpected row counts. Full-resolution composites are generated only with `--save-composites` and remain ignored.
