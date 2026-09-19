"""
Run one full rule-based-heuristic episode, headless, fully instrumented.

Calls the SAME action_decision() heuristic used by agent_server.py /
local_playground.py, in-process (no HTTP round trip) -- this matches how the
real evaluator drives agents (one action per agent per tick) but skips network
overhead so sweeps run faster. See training/rule_based/README.md for why this
is a fair stand-in for throughput/behavior purposes.

Usage:
    python3 training/rule_based/run_episode.py --seed 1 --max-sim-time 3000
"""
import argparse
import json
import os
import random
import sys
import time as wallclock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd

from src.core import SimulationCore
from src.utils.controllers.heuristic_policy import action_decision
from training.rule_based.instrumentation import (
    RunLog,
    attach_instrumentation,
    patch_environment_for_headless,
    record_tick,
)


def run_episode(seed: int, max_sim_time: float = 3000.0, dt: float = 1 / 10, verbose: bool = False) -> dict:
    """
    Run one episode to extinction or max_sim_time, whichever comes first.

    Returns a dict with:
      summary: dict of scalar outcomes for this run (final score, extinction time, etc.)
      ticks:   DataFrame, one row per simulated tick (population, energy, score components)
      deaths:  DataFrame, one row per agent death (cause, energy/age at death)
      births:  DataFrame, one row per successful reproduction
    """
    patch_environment_for_headless()  # skip the unused per-pixel biome render; see instrumentation.py
    sim = SimulationCore(seed=seed, dt=dt)
    log = RunLog()
    attach_instrumentation(sim.env, log)

    action_rng = random.Random(seed)
    actions = []
    prev_score = sim.env.score
    wall_t0 = wallclock.time()

    while sim.env.time <= max_sim_time and len(sim.env.agents) > 0:
        prev_death_count = len(log.deaths)
        state = sim.step(actions)

        # Deaths recorded during THIS call to sim.step are exactly the tail of
        # log.deaths appended since prev_death_count -- attach_instrumentation's
        # kill_agent hook fires synchronously inside sim.step().
        new_deaths = log.deaths[prev_death_count:]
        predation_penalty = sum(d["energy"] / 100.0 for d in new_deaths if d["cause"] == "predator")
        prev_score = record_tick(sim.env, log, prev_score, predation_penalty, dt)

        actions = []
        for agent, agent_state in zip(sim.env.agents, state["observations"]):
            action = action_decision(agent_state, action_rng)
            actions.append((agent.agent_id, action))

        if verbose and int(sim.env.time * 10) % 500 == 0:
            print(f"  t={sim.env.time:7.1f}  score={state['score']:8.2f}  agents={state['num_agents']:3d}")

    wall_elapsed = wallclock.time() - wall_t0
    extinct = len(sim.env.agents) == 0

    cause_counts = pd.Series([d["cause"] for d in log.deaths]).value_counts().to_dict() if log.deaths else {}

    summary = {
        "seed": seed,
        "final_score": sim.env.score,
        "extinct": extinct,
        "extinction_time": sim.env.time if extinct else None,
        "final_sim_time": sim.env.time,
        "final_num_agents": len(sim.env.agents),
        "wall_seconds": wall_elapsed,
        "sim_seconds_per_wall_second": (sim.env.time / wall_elapsed) if wall_elapsed > 0 else None,
        "num_deaths": len(log.deaths),
        "num_births": len(log.births),
        "deaths_predator": int(cause_counts.get("predator", 0)),
        "deaths_starvation_old": int(cause_counts.get("starvation_old", 0)),
        "deaths_starvation_young": int(cause_counts.get("starvation_young", 0)),
    }
    return {
        "summary": summary,
        "ticks": pd.DataFrame(log.ticks),
        "deaths": pd.DataFrame(log.deaths),
        "births": pd.DataFrame(log.births),
    }


def save_run(result: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    result["ticks"].to_csv(os.path.join(out_dir, "ticks.csv"), index=False)
    result["deaths"].to_csv(os.path.join(out_dir, "deaths.csv"), index=False)
    result["births"].to_csv(os.path.join(out_dir, "births.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(result["summary"], f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run one instrumented rule-based-heuristic episode.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-sim-time", type=float, default=3000.0)
    parser.add_argument("--out", type=str, default=None, help="Output dir; default training/rule_based/results/run_seed<seed>")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run_episode(seed=args.seed, max_sim_time=args.max_sim_time, verbose=args.verbose)
    print(json.dumps(result["summary"], indent=2))

    out_dir = args.out or os.path.join(os.path.dirname(__file__), "results", f"run_seed{args.seed}")
    save_run(result, out_dir)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
