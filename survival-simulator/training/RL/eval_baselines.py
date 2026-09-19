"""
Scripted-policy baselines for one curriculum stage (stage 1 or stage 2 etc.),
to answer "what survive-rate is actually achievable here?" before judging the
RL policy or picking a goal threshold.

Policies:
  do_nothing      stand still (pure energy drain). Floor.
  random          uniform random actions in the same normalized action space
                  the RL policy uses.
  heuristic       src/utils/controllers/heuristic_policy.py's action_decision
                  (observation-only, exactly what a deployed agent sees;
                  includes its own predator-evasion rules).
  oracle_nearest  PRIVILEGED forager: sees every fruit on the map, walks to the
                  nearest one at walking speed; if none exist, walks to the
                  nearest tree older than 20s, else stands still. Ignores
                  predators entirely.
  oracle_value    PRIVILEGED forager: picks the fruit with the best
                  (energy - walking cost) per unit of travel time.
  oracle_evade    PRIVILEGED: oracle_nearest foraging, but if an awake
                  predator is within EVADE_RADIUS (true distance) it sprints
                  straight away from it instead.
  approach        PRIVILEGED, diagnostic only: walks toward the nearest
                  predator (worst-case behaviour; used by reward_sanity.py).

The oracles are greedy and ignore obstacles (the sim deflects collisions), so
they are a strong reference, NOT a proof of the true optimum.

Default policy set: stage with no predators -> do_nothing, random, heuristic,
oracle_nearest, oracle_value. Stage with predators -> do_nothing, random,
heuristic, oracle_nearest, oracle_evade. Override with --policies.

--mask-spawn simulates the fix for the "blocked reproduction still costs 100 energy" issue
(stages 1-3). "wasted_spawn/ep" = spawn requests per episode that cost 100 energy (or would
have, when masked) without creating a child. See spawn_hook.py.

Death causes are counted exactly (kill_agent with energy > 0 = predator,
energy <= 0 = starvation; same trick as reward.py / rule_based instrumentation).
"ext_by_pred%" = share of EXTINCT episodes whose last death was a predator kill.

Uses the same CurriculumEnv (same patches, same stage config, same episode-seed
scheme) as training and eval_checkpoints.py. With the default --seed-base
(777000) and CHUNK=10, the first 100 episodes use the SAME maps as pass 1 of
eval_checkpoints.py, so numbers are directly comparable (per stage).

Metric: survive = >=1 agent alive when the stage's max_sim_time is reached.

Usage (repo root, venv active):
    python3 -u training/RL/eval_baselines.py --stage stage2_single_predator_evasion --episodes 200 --workers 14
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import random
import sys
import time
import types
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
from training.RL.spawn_hook import hook_spawn

CHUNK = 10
POLICIES = ("do_nothing", "random", "heuristic", "oracle_nearest", "oracle_value", "oracle_evade", "approach")
DEFAULT_NO_PRED = ("do_nothing", "random", "heuristic", "oracle_nearest", "oracle_value")
DEFAULT_PRED = ("do_nothing", "random", "heuristic", "oracle_nearest", "oracle_evade")
WALK_COST = 0.05  # energy per unit distance while walking (environment.py)
EVADE_RADIUS = 110.0  # oracle_evade flees awake predators closer than this (true distance)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _go_to(agent, tx, ty, sprint=False):
    dx, dy = tx - agent.x, ty - agent.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return _idle(agent)
    rel = _wrap(math.atan2(dy, dx) - agent.direction)
    top = agent.sprint_speed if sprint else agent.speed
    return ActionRequest(agent_id=agent.agent_id, move_distance=min(top, dist),
                         move_direction=rel, turn_angle=0.0, spawn_agent=False)


def _flee_from(agent, px, py):
    dx, dy = agent.x - px, agent.y - py
    rel = _wrap(math.atan2(dy, dx) - agent.direction)
    return ActionRequest(agent_id=agent.agent_id, move_distance=agent.sprint_speed,
                         move_direction=rel, turn_angle=0.0, spawn_agent=False)


def _idle(agent):
    return ActionRequest(agent_id=agent.agent_id, move_distance=0.0, move_direction=0.0,
                         turn_angle=0.0, spawn_agent=False)


def _nearest(items, agent):
    return min(items, key=lambda o: (o.x - agent.x) ** 2 + (o.y - agent.y) ** 2)


def _oracle_forage(env, agent, dt, mode):
    fruits = env.fruits
    if fruits:
        if mode == "oracle_value":
            def value(f):
                d = math.hypot(f.x - agent.x, f.y - agent.y)
                gain = f.energy - WALK_COST * d
                t = d * dt / max(agent.speed, 1e-6)  # seconds of walking
                return gain / (t + 0.5)
            f = max(fruits, key=value)
        else:
            f = _nearest(fruits, agent)
        return _go_to(agent, f.x, f.y)
    trees = [t for t in env.trees if t.age >= 20]
    if trees:
        t = _nearest(trees, agent)
        return _go_to(agent, t.x, t.y)
    return _idle(agent)


def _oracle_action(env, agent, dt, mode):
    if mode == "oracle_evade":
        awake = [p for p in env.predators if not getattr(p, "resting", False)]
        if awake:
            p = _nearest(awake, agent)
            if math.hypot(p.x - agent.x, p.y - agent.y) < EVADE_RADIUS:
                return _flee_from(agent, p.x, p.y)
        return _oracle_forage(env, agent, dt, "oracle_nearest")
    if mode == "approach":
        if env.predators:
            p = _nearest(env.predators, agent)
            return _go_to(agent, p.x, p.y)
        return _idle(agent)
    return _oracle_forage(env, agent, dt, mode)


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


def _hook_death_causes(env, counts):
    """Count deaths by cause on this env instance (exact: energy>0 at kill time = predator)."""
    orig = env.kill_agent

    def hook(self, agent):
        cause = "pred" if agent.energy > 0 else "starve"
        counts[cause] += 1
        counts["last"] = cause
        return orig(agent)

    env.kill_agent = types.MethodType(hook, env)


def _run_chunk(job):
    policy, seed, n_eps, stage_name, mask_spawn = job
    stage = get_stage(stage_name)
    cenv = CurriculumEnv(stage, seed=seed)
    rng = random.Random(seed)
    out = []
    for _ in range(n_eps):
        cenv.reset()
        sim = cenv.sim
        env = sim.env
        counts = {"pred": 0, "starve": 0, "last": None}
        _hook_death_causes(env, counts)
        spawn_counts = {"wasted": 0}
        hook_spawn(env, spawn_counts, mask_spawn)
        start_agents = len(env.agents)
        while True:
            actions = _actions(policy, env, sim.dt, rng)
            begin_tick(env, cenv.reward_state)
            sim.step(actions)
            compute_rewards(env, cenv.reward_state, sim.dt)
            if len(env.agents) == 0 or env.time >= stage.max_sim_time:
                break
        out.append((policy, float(env.time), len(env.agents), start_agents,
                    counts["pred"], counts["starve"], counts["last"] or "", spawn_counts["wasted"]))
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
    ap.add_argument("--policies", default=None, help="comma list; default depends on whether the stage has predators")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed-base", type=int, default=777000)
    ap.add_argument("--mask-spawn", action="store_true",
                    help="Simulate the FIX for the blocked-reproduction energy charge: force spawn_agent=False "
                         "(stages 1-3 only). Without it, blocked spawn requests still cost 100 energy (current behaviour).")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "eval_results"))
    args = ap.parse_args()

    stage = get_stage(args.stage)
    if args.policies:
        policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    else:
        policies = list(DEFAULT_NO_PRED if stage.predators == "off" else DEFAULT_PRED)
    bad = [p for p in policies if p not in POLICIES]
    if bad:
        sys.exit(f"unknown policies: {bad}; choose from {POLICIES}")

    n_chunks = math.ceil(args.episodes / CHUNK)
    if args.mask_spawn and stage.reproduction:
        sys.exit(f"--mask-spawn is only meaningful for stages with reproduction disabled; {args.stage} has it enabled.")
    jobs = [(p, args.seed_base + c, CHUNK, args.stage, args.mask_spawn) for p in policies for c in range(n_chunks)]
    print(f"{len(jobs)} jobs ({len(policies)} policies x {n_chunks} chunks x {CHUNK} eps), {args.workers} workers", flush=True)

    res = defaultdict(list)
    t0 = time.time()
    ctx = mp.get_context("spawn") if sys.platform == "win32" else mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_run_chunk, jobs), 1):
            for row in chunk:
                res[row[0]].append(row[1:])
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"[{i}/{len(jobs)}] elapsed={el/60:.1f}m eta={el/i*(len(jobs)-i)/60:.1f}m", flush=True)

    print(f"\n=== Baselines on {args.stage} (cap {stage.max_sim_time:.0f}s, {args.episodes} eps each, "
          f"spawn charge {'MASKED (fixed behaviour)' if args.mask_spawn else 'ACTIVE (current behaviour)'}) ===")
    print(f"{'policy':<15} {'survive%':>8} {'95% CI':>10} {'all-alive%':>10} {'mean_t':>7} {'median_t':>8} "
          f"{'mean_alive':>10} {'pred_d/ep':>9} {'starv_d/ep':>10} {'ext_by_pred%':>12} {'wasted_spawn/ep':>15}")
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw = os.path.join(args.out, f"{args.stage}_baselines_raw_{stamp}.csv")
    with open(raw, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "sim_time", "final_agents", "start_agents", "predator_deaths", "starvation_deaths", "last_death_cause", "wasted_spawn"])
        for p in policies:
            eps = res[p]
            n = len(eps)
            k = sum(1 for e in eps if e[1] > 0)
            k_all = sum(1 for e in eps if e[1] == e[2])
            lo, hi = wilson(k, n)
            ts = [e[0] for e in eps]
            extinct = [e for e in eps if e[1] == 0]
            ext_pred = 100 * sum(1 for e in extinct if e[5] == "pred") / len(extinct) if extinct else float("nan")
            print(f"{p:<15} {100*k/n:8.1f} [{100*lo:3.0f},{100*hi:3.0f}] {100*k_all/n:10.1f} "
                  f"{np.mean(ts):7.1f} {np.median(ts):8.1f} {np.mean([e[1] for e in eps]):10.2f} "
                  f"{np.mean([e[3] for e in eps]):9.2f} {np.mean([e[4] for e in eps]):10.2f} {ext_pred:12.1f} "
                  f"{np.mean([e[6] for e in eps]):15.2f}")
            for t, a, s, pd, sd, last, wsp in eps:
                w.writerow([p, t, a, s, pd, sd, last, wsp])
    print(f"\nRaw episodes: {raw}")


if __name__ == "__main__":
    main()
