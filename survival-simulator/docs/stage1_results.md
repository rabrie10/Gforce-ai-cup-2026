> **Superseded, kept for history.** The "RL iter400 (~70%) roughly equals
> the heuristic (70%)" conclusion below was measured *before* a bug fix:
> the environment was charging an agent's 100-energy spawn cost even when
> the curriculum blocked the spawn outright (reproduction is disabled on
> stages 1-3). After removing that blocked-spawn energy charge, the
> heuristic's actual stage-1 survival is 99.5-100%, not 70% -- the RL
> policy here does **not** match heuristic performance, it falls well
> short of it. The "90% goal is unrealistic / the oracle is a ceiling"
> conclusion below is also wrong for the same reason: with the bug fixed,
> the heuristic itself clears the original 90% goal. See the top-level
> [README](../README.md#our-solution) for the final approach and results
> actually used for submission.

---

# Stage 1 (foraging) results, 2026-09-19

Metric: survive = >=1 of 3 agents alive at the 300s cap (curriculum.py stage-1 goal). Original goal: 90% over a 50-episode window.

## Training run (2000 iterations, hit the --iterations cap; goal never reached)
- Best 100-iteration windows: ~40% extinction (~60% survive) at iters 401-500 and 1201-1300.
- Regressed after ~1300: extinction rose to ~79% by iters 1901-2000. `stage1_foraging_final.pt` is much worse than mid-run checkpoints.
- Training is unstable: iter450 scored 13% / 5% (stoch/det) while iter440 and iter460 scored 44-61%.
- `fruit_energy_consumed` undercounts long episodes (accumulator resets every 512-tick rollout; a 300s episode is 3000 ticks). Do not trust its zero-fruit share.

## Checkpoint evaluation (eval_checkpoints.py, paired seeds, 400 eps each for the top 4)
| checkpoint | stoch survive | det survive |
|---|---|---|
| iter400 | 70.5% [66,75] | 69.8% [65,74] |
| iter1220 | 63.7% | 68.8% |
| iter1280 | 57.0% | 72.0% |
| iter1240 | 61.8% | 62.0% |
| final (100 eps) | 30% | 63% |

Chosen: iter400 (copied to stage1_foraging_best.pt). Not clearly better than iter1220/1280 in det mode; wins on consistency across both modes. Slightly optimistic due to pass-1 selection.

## Scripted baselines (eval_baselines.py, 200 eps each, same seeds as checkpoint pass 1)
| policy | survive | 95% CI | mean t |
|---|---|---|---|
| do_nothing | 0% | [0,2] | 171s |
| random | 0% | [0,2] | 7.5s |
| heuristic | 70.0% | [63,76] | 268s |
| oracle_nearest (privileged, greedy) | 86.5% | [81,91] | 277s |
| oracle_value (privileged, greedy) | 87.0% | [82,91] | 275s |

Only ~1 agent survives on average (all-3-alive is 1.5-2.5%).

## Conclusions
- RL iter400 (~70%) roughly equals the heuristic (70%); it does not beat it.
- The 90% goal sits at/above what even a privileged greedy oracle achieves (~87%), so it is not a realistic target for an observation-only policy. Use a threshold of about 0.75-0.80 at most, or treat stage 1 as done.
- Open: training instability (LR/entropy/value drift), best-checkpoint saving, per-iteration entropy/KL logging.
