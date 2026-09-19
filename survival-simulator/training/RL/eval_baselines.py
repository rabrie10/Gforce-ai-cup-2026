"""
Scripted-policy baselines for one curriculum stage (default: stage1_foraging),
to answer "what survive-rate is actually achievable here?" before judging the
RL policy or picking a goal threshold.

Policies:
  do_nothing      stand still (pure energy drain). Floor: shows how much
                  foraging matters at all.
  random          uniform random actions in the same normalized action space
                  the RL policy uses.
  heuristic       src/utils/controllers/heuristic_policy.py's action_decision
                  (observation-only, exactly what a deployed agent sees).
  oracle_nearest  PRIVILEGED: sees every fruit on the map, walks to the
                  nearest one at walking speed; if none exist, walks to the
                  nearest tree older than 20s (trees only fruit at age>=20),
                  else stands still.
  oracle_value    PRIVILEGED: like oracle_nearest but picks the fruit with the
                  best (energy - walking cost) per unit of travel time.

The oracles are greedy and ignore obstacles (the sim deflects collisions), so
they are a strong reference, NOT a proof of the true optimum.

Uses the same CurriculumEnv (same patches, same stage config, same episode-seed
scheme) as training and eval_checkpoints.py. With the default --seed-base
(777000) and CHUNK=10, the first 100 episodes use the SAME maps as pass 1 of
eval_checkpoints.py, so the numbers are directly comparable.

Metric: survive = >=1 agent alive when the stage's max_sim_time is reached
(same as the stage-1 goal in curriculum.py).

Usage (repo root, venv active):
    python3 -u training/RL/eval_baselines.py --stage stage1_foraging --episodes 200 --workers 14
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from src.utils.DTOs import ActionRequest
from src.utils.controllers.heuristic_policy import action_decision as heuristic_action
from training.RL.curriculum import get_stage
from training.RL.env_wrapper import CurriculumEnv, decode_action
from training.RL.reward import begin_tick, compute_rewards

CHUNK = 10
POLICIES = ("do_nothing", "random", "heuristic", "oracle_nearest", "oracle_value")
WALK_COST = 0.05  # energy per unit distance while walking (environment.py)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _go_to(agent, tx, ty):
    dx, dy = tx - agent.x, ty - agent.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return ActionRequest(agent_id=agent.agent_id, move_distance=0.0, move_direction=0.0,
                             turn_angle=0.0, spawn_agent=False)
    rel = _wrap(math.atan2(dy, dx) - agent.direction)
    return ActionRequest(agent_id=agent.agent_id, move_distance=min(agent.speed, dist),
                         move_direction=rel, turn_angle=0.0, spawn_agent=False)


def _idle(agent):
    return ActionRequest(agent_id=agent.agent_id, move_distance=0.0, move_direction=0.0,
                         turn_angle=0.0, spawn_agent=False)


def _oracle_action(env, agent, dt, mode):
    fruits = env.fruits
    if fruits:
        if mode == "oracle_nearest":
            f = min(fruits, key=lambda f: (f.x - agent.x) ** 2 + (f.y - agent.y) ** 2)
        else:
            def value(f):
                d = math.hypot(f.x - agent.x, f.y - agent.y)
                gain = f.energy - WALK_COST * d
                t = d * dt / max(agent.speed, 1e-6)  # seconds of walking
                return gain / (t + 0.5)
            f = max(fruits, key=value)
        return _go_to(agent, f.x, f.y)
    trees = [t for t in env.trees if t.age >= 20]
    if trees:
        t = min(trees, key=lambda t: (t.x - agent.x) ** 2 + (t.y - agent.y) ** 2)
        return _go_to(agent, t.x, t.y)
    return _idle(agent)


def _actions(policy, env, dt, rng):
    out = []
    for agent in env.agents:
        if policy == "do_nothing":
            a = _idle(agent)
        elif policy == "random":
            a = decode_action(agent.agent_id, np.array([rng.uniform(-1, 1) for _ in range(4)]), agent.sprint_speed)
        elif policy == "heuristic":
            a = heuristic_action(env.get_agent_state(agent.agent_id), rng)
        else:
            a = _oracle_action(env, agent, dt, policy)
        out.append((agent.agent_id, a))
    return out


def _run_chunk(job):
    policy, seed, n_eps, stage_name = job
    stage = get_stage(stage_name)
    cenv = CurriculumEnv(stage, seed=seed)
    rng = random.Random(seed)
    out = []
    for _ in range(n_eps):
        cenv.reset()
        sim = cenv.sim
        env = sim.env
        start_agents = len(env.agents)
        while True:
            actions = _actions(policy, env, sim.dt, rng)
            begin_tick(env, cenv.reward_state)
            sim.step(actions)
            compute_rewards(env, cenv.reward_state, sim.dt)
            if len(env.agents) == 0 or env.time >= stage.max_sim_time:
                break
        out.append((policy, float(env.time), len(env.agents), start_agents))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage1_foraging")
    ap.add_argument("--episodes", type=int, default=200, help="episodes per policy")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed-base", type=int, default=777000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "eval_results"))
    args = ap.parse_args()

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    bad = [p for p in policies if p not in POLICIES]
    if bad:
        sys.exit(f"unknown policies: {bad}; choose from {POLICIES}")

    n_chunks = math.ceil(args.episodes / CHUNK)
    jobs = [(p, args.seed_base + c, CHUNK, args.stage) for p in policies for c in range(n_chunks)]
    print(f"{len(jobs)} jobs ({len(policies)} policies x {n_chunks} chunks x {CHUNK} eps), {args.workers} workers", flush=True)

    res = defaultdict(list)
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_run_chunk, jobs), 1):
            for p, t, alive, start in chunk:
                res[p].append((t, alive, start))
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"[{i}/{len(jobs)}] elapsed={el/60:.1f}m eta={el/i*(len(jobs)-i)/60:.1f}m", flush=True)

    stage = get_stage(args.stage)
    print(f"\n=== Baselines on {args.stage} (cap {stage.max_sim_time:.0f}s, {args.episodes} eps each) ===")
    print(f"{'policy':<15} {'survive%':>8} {'95% CI':>13} {'all-alive%':>10} {'mean_t':>7} {'median_t':>8} {'mean_alive':>10}")
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw = os.path.join(args.out, f"{args.stage}_baselines_raw_{stamp}.csv")
    with open(raw, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "sim_time", "final_agents", "start_agents"])
        for p in policies:
            eps = res[p]
            n = len(eps)
            k = sum(1 for _, a, _ in eps if a > 0)
            k_all = sum(1 for _, a, s in eps if a == s)
            lo, hi = wilson(k, n)
            ts = [t for t, _, _ in eps]
            print(f"{p:<15} {100*k/n:8.1f} [{100*lo:4.0f},{100*hi:4.0f}] {100*k_all/n:10.1f} "
                  f"{np.mean(ts):7.1f} {np.median(ts):8.1f} {np.mean([a for _, a, _ in eps]):10.2f}")
            for t, a, s in eps:
                w.writerow([p, t, a, s])
    print(f"\nRaw episodes: {raw}")


if __name__ == "__main__":
    main()
