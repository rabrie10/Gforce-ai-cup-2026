# rule_based/

Instrumentation, multi-seed sweeps, and visualization for the hand-tuned
heuristic in `src/utils/controllers/heuristic_policy.py`. Nothing in this
folder modifies any file under `src/` -- everything attaches to a running
`Environment` instance from the outside. See the docstring at the top of
`instrumentation.py` for exactly how and why that's safe.

The point of this folder is to answer, with data instead of guesses: what
actually kills the colony (predators, starvation, old age), and how much
does score really depend on each of the three things it's built from. That
result is both the rule-based baseline the RL work in `../RL/` has to beat,
and the evidence that should drive curriculum design later.

## Files

- `instrumentation.py` -- monkeypatches `kill_agent` / `spawn_agent` on one
  Environment instance to log every death (with an exact, not inferred,
  cause: predator / starved-young / starved-past-max-age) and birth, plus a
  per-tick sampler that decomposes `env.score`'s delta into its three
  additive terms (time survived, fruit eaten, predation penalty) without
  needing to hook fruit-eating at all -- see the module docstring for the
  derivation.
- `run_episode.py` -- runs one full episode headless (`SDL_VIDEODRIVER=dummy`,
  no window), calling the existing `action_decision()` heuristic in-process
  per tick, exactly like `local_playground.py` does. Saves `ticks.csv`,
  `deaths.csv`, `births.csv`, `summary.json` per run.
- `sweep.py` -- runs many seeds in parallel (each seed is a fully
  independent episode, so this uses `multiprocessing`), writes
  `results/sweep_summary.csv` plus one folder per seed under `results/`.
- `visualize.py` -- reads a completed sweep and writes PNGs to
  `results/plots/`: extinction-time distribution, population-over-time
  (all seeds overlaid + mean), death-cause breakdown, score decomposed into
  its three components for a representative run, and mean colony energy
  over time.

## Running it

From the repo root, with the project's own dependencies available
(pygame, shapely, scipy, fastapi, pydantic; these already exist in the
project's own virtual environment per `requirements.txt`):

```
# Quick smoke test (one seed, short run):
python3 training/rule_based/run_episode.py --seed 1 --max-sim-time 120 --verbose

# Full sweep (defaults to seeds 1-30, full 3000s runs, parallel across cores):
python3 training/rule_based/sweep.py --seeds 1-30 --max-sim-time 3000

# Charts, once a sweep has produced results/sweep_summary.csv:
python3 training/rule_based/visualize.py
```

A note on cost: on this machine, a full 3000s run takes roughly a few
minutes headless (see `summary.json`'s `sim_seconds_per_wall_second` for the
measured rate on your actual hardware -- it varies with population size and
CPU). `sweep.py` defaults to `cpu_count() - 1` parallel workers so a 30-seed
sweep doesn't run sequentially; tune `--workers` down if you want to keep
using the machine for something else while it runs, or up if you don't.

## Why the death-cause split can be trusted

Short version, full derivation in `instrumentation.py`'s module docstring:
`kill_agent()` has exactly two call sites in `environment.py`, and they're
mutually exclusive within a tick by construction (a starved agent is removed
from the grid before the predator loop runs, so the predator loop can only
ever kill an agent with positive energy). So `agent.energy <= 0` at the
moment `kill_agent` fires is not a heuristic guess about cause, it's the
literal condition the source code just branched on.

## Next steps (not yet built)

- Parameter search (random search or CMA-ES) over the heuristic's tunable
  constants (`PREDATOR_CHARGE_DISTANCE`, `SPAWN_ENERGY_FRACTION`,
  `EXPLORE_TURN_RANGE`, `EDGE_AVOID_DISTANCE`), using mean/median
  extinction time across many seeds as the fitness -- once the death-cause
  breakdown says which failure mode is actually worth tuning against.
