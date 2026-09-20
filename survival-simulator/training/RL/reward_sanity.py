"""
Reward sanity check for the evasion-shaping term (w_danger), NO training.

Runs scripted policies on a predator stage and measures what the shaped reward
actually looks like tick by tick, so the design (w_danger, DANGER_RADIUS) can be
judged from data before spending training time:

  * Does shaping give the right per-tick CREDIT? An "encounter branching" test
    replays the same seed up to the first moment an awake predator is perceived
    at 60-130 units, then branches: flee / walk away / stand still / approach /
    heuristic, and measures the shaping reward paid in the first ticks after the
    branch (flee should be paid positive immediately, approach negative).
    Note: total shaping over an episode telescopes to endpoints only, so dying
    and escaping have the SAME net shaping -- that is expected; the -w_death
    term separates them. What shaping must provide is EARLY, per-tick credit.
  * How big is it next to the other per-tick reward terms?
  * How often does Phi jump discontinuously (a predator "pops" into or out of
    the agent's own perception), which pays/charges a lump-sum shaping reward
    for merely seeing or losing sight of a predator?
  * What is the death-step refund (Phi(terminal)=0 convention) and the
    resulting effective death penalty?

Shaping for a tick = (reward with w_danger) - (reward with w_danger=0), computed by
calling reward.compute_rewards twice per tick with different weights (it has no
side effect on the per-tick reward inputs, only on the fruit diagnostic).

Policies come from eval_baselines.py (same seeds/maps as everything else):
  do_nothing, approach (walk at nearest predator), oracle_evade (privileged
  flee/forage), heuristic (its own evasion rules), oracle_nearest (forages,
  ignores predators).

--w-danger, --danger-radius and --phi-mode override the stage's / default values, so you can compare
alternative shaping settings on identical behaviour without training anything.

Usage (repo root, venv active):
    python3 -u training/RL/reward_sanity.py --stage stage2_single_predator_evasion --episodes 60 --workers 6
"""
import argparse
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

import training.RL.reward as R
from training.RL.curriculum import get_stage
from training.RL.env_wrapper import CurriculumEnv
from training.RL.eval_baselines import POLICIES, _actions, wilson
from training.RL.reward import begin_tick, compute_rewards

CHUNK = 5
DEFAULT_POLICIES = ("do_nothing", "approach", "oracle_evade", "heuristic", "oracle_nearest")
JUMP = 0.3  # |delta Phi| in ONE tick counted as a "discontinuity"


def _run_chunk(job):
    policy, seed, n_eps, stage_name, w_danger_override, radius_override, phi_mode = job
    if radius_override is not None:
        R.DANGER_RADIUS = float(radius_override)
    R.PHI_MODE = phi_mode
    stage = get_stage(stage_name)
    cenv = CurriculumEnv(stage, seed=seed)
    rng = random.Random(seed)
    rs = cenv.reward_state
    full_w = dict(rs.weights)
    if w_danger_override is not None:
        full_w["w_danger"] = float(w_danger_override)
    base_w = dict(full_w, w_danger=0.0)
    w_death = full_w["w_death"]
    w_danger = full_w["w_danger"]

    episodes = []
    for _ in range(n_eps):
        cenv.reset()
        sim = cenv.sim
        env = sim.env
        dt = sim.dt
        rs.weights = full_w
        start_agents = len(env.agents)

        ticks_alive = 0            # agent-ticks for agents alive at end of tick
        ticks_perceived = 0        # ... of which a predator was perceived before or after
        abs_base_alive = 0.0       # sum |base reward| over alive agent-ticks
        sh_perceived = []          # per perceived alive agent-tick shaping (signed)
        sh_total = 0.0             # all shaping reward (alive ticks + death steps)
        base_total = 0.0           # all non-shaping reward (incl. death penalty)
        pop_in = pop_out = 0
        pop_in_mag = []
        deaths = 0
        deaths_perceived = 0
        refunds = []

        while True:
            actions = _actions(policy, env, dt, rng)
            begin_tick(env, rs)
            sim.step(actions)
            rs.weights = base_w
            base = compute_rewards(env, rs, dt)
            rs.weights = full_w
            total = compute_rewards(env, rs, dt)
            by_id = {a.agent_id: a for a in env.agents}
            living = set(by_id)

            for aid, r_tot in total.items():
                r_base = base.get(aid, 0.0)
                sh = r_tot - r_base
                phi_prev = rs.prev_phi.get(aid, 0.0)
                sh_total += sh
                base_total += r_base
                if aid in living:
                    phi_curr = R._phi(env, by_id[aid], env.agent_observations.get(aid, []))
                    ticks_alive += 1
                    abs_base_alive += abs(r_base)
                    if phi_prev < 0 or phi_curr < 0:
                        ticks_perceived += 1
                        sh_perceived.append(sh)
                    if phi_prev == 0 and phi_curr < -JUMP:
                        pop_in += 1
                        pop_in_mag.append(-phi_curr)
                    elif phi_prev < -JUMP and phi_curr == 0:
                        pop_out += 1
                else:
                    deaths += 1
                    if phi_prev < 0:
                        deaths_perceived += 1
                        refunds.append(-phi_prev)

            if len(env.agents) == 0 or env.time >= stage.max_sim_time:
                break

        episodes.append(dict(
            t=float(env.time), alive=len(env.agents), start=start_agents,
            ticks_alive=ticks_alive, ticks_perceived=ticks_perceived,
            abs_base_alive=abs_base_alive, sh_perceived=np.asarray(sh_perceived, dtype=np.float32),
            sh_total=sh_total, base_total=base_total, pop_in=pop_in, pop_out=pop_out,
            pop_in_mag=pop_in_mag, deaths=deaths, deaths_perceived=deaths_perceived, refunds=refunds,
        ))
    return policy, episodes, w_death, w_danger


ENC_MIN, ENC_MAX = 60.0, 130.0   # perceived predator distance window that defines an "encounter"
HORIZONS = (5, 20)               # ticks after the branch over which credit is measured
BRANCHES = ("flee", "walk_away", "still", "approach", "heuristic")


def _step_both(cenv, actions, base_w, full_w):
    env, sim, rs = cenv.sim.env, cenv.sim, cenv.reward_state
    begin_tick(env, rs)
    sim.step(actions)
    rs.weights = base_w
    base = compute_rewards(env, rs, sim.dt)
    rs.weights = full_w
    total = compute_rewards(env, rs, sim.dt)
    return base, total


def _awake_predators(env):
    return [p for p in env.predators if not getattr(p, "resting", False)]


def _branch_actions(branch, env, rng):
    """Same action for every living agent, chosen by `branch`."""
    from training.RL.eval_baselines import _flee_from, _go_to, _idle, _nearest, heuristic_action
    out = []
    for agent in env.agents:
        preds = env.predators
        if branch == "still" or not preds:
            a = _idle(agent)
        elif branch == "heuristic":
            a = heuristic_action(env.get_agent_state(agent.agent_id), rng)
        else:
            p = _nearest(preds, agent)
            if branch == "flee":
                a = _flee_from(agent, p.x, p.y)
            elif branch == "walk_away":
                a = _flee_from(agent, p.x, p.y)
                a.move_distance = agent.speed
            else:  # approach
                a = _go_to(agent, p.x, p.y)
        out.append((agent.agent_id, a))
    return out


def _encounter_job(job):
    seed, stage_name, w_danger_override, radius_override, phi_mode = job
    if radius_override is not None:
        R.DANGER_RADIUS = float(radius_override)
    R.PHI_MODE = phi_mode
    from training.RL.eval_baselines import _actions
    stage = get_stage(stage_name)

    def fresh():
        c = CurriculumEnv(stage, seed=seed)
        c.reset()
        full = dict(c.reward_state.weights)
        if w_danger_override is not None:
            full["w_danger"] = float(w_danger_override)
        c.reward_state.weights = full
        return c, full, dict(full, w_danger=0.0)

    # --- discovery run: forage with oracle_nearest until an encounter appears
    cenv, full_w, base_w = fresh()
    env, dt = cenv.sim.env, cenv.sim.dt
    rng = random.Random(seed)
    found = None
    max_ticks = int(stage.max_sim_time / dt)
    for k in range(max_ticks):
        awake = _awake_predators(env)
        if awake:
            for a in env.agents:
                obs = env.agent_observations.get(a.agent_id, [])
                pd = [o["distance"] for o in obs if o.get("type") == "Predator"]
                if pd and ENC_MIN <= min(pd) <= ENC_MAX and any(
                        math.hypot(p.x - a.x, p.y - a.y) <= ENC_MAX + 20 for p in awake):
                    found = (k, a.agent_id, min(pd), (a.x, a.y))
                    break
        if found or len(env.agents) == 0:
            break
        _step_both(cenv, _actions("oracle_nearest", env, dt, rng), base_w, full_w)
    if not found:
        return seed, None, 0

    k, aid, d0, pos0 = found
    mismatches = 0
    res = {}
    for branch in BRANCHES:
        c2, fw, bw = fresh()
        e2, dt2 = c2.sim.env, c2.sim.dt
        r2 = random.Random(seed)
        for _ in range(k):
            _step_both(c2, _actions("oracle_nearest", e2, dt2, r2), bw, fw)
        ag = e2.agents_dict.get(aid)
        if ag is None or abs(ag.x - pos0[0]) > 1e-6 or abs(ag.y - pos0[1]) > 1e-6:
            mismatches += 1
        sh_alive = [0.0] * max(HORIZONS)   # shaping while agent alive after the step
        sh_all = [0.0] * max(HORIZONS)     # including the death step
        ret_all = [0.0] * max(HORIZONS)    # total reward incl. death penalty
        died_at = None
        for t in range(max(HORIZONS)):
            if aid not in e2.agents_dict:
                break
            base, total = _step_both(c2, _branch_actions(branch, e2, r2), bw, fw)
            sh = total.get(aid, 0.0) - base.get(aid, 0.0)
            sh_all[t] = sh
            ret_all[t] = total.get(aid, 0.0)
            if aid in e2.agents_dict:
                sh_alive[t] = sh
            else:
                died_at = t
        res[branch] = dict(
            **{f"sh_alive_{h}": sum(sh_alive[:h]) for h in HORIZONS},
            **{f"sh_all_{h}": sum(sh_all[:h]) for h in HORIZONS},
            **{f"ret_{h}": sum(ret_all[:h]) for h in HORIZONS},
            **{f"died_{h}": float(died_at is not None and died_at < h) for h in HORIZONS},
        )
    return seed, dict(d0=d0, res=res), mismatches


def _run_encounter_test(args, w_danger, radius):
    seeds = [args.seed_base + i for i in range(args.encounters)]
    jobs = [(sd, args.stage, args.w_danger, args.danger_radius, args.phi_mode) for sd in seeds]
    print(f"\n--- Encounter branching test: up to {len(jobs)} seeds (first awake predator perceived at "
          f"{ENC_MIN:.0f}-{ENC_MAX:.0f} units, then every agent follows the branch policy) ---", flush=True)
    ctx = mp.get_context("spawn") if sys.platform == "win32" else mp.get_context("fork")
    rows = []
    mism = 0
    with ctx.Pool(args.workers) as pool:
        for i, (sd, out, mm) in enumerate(pool.imap_unordered(_encounter_job, jobs), 1):
            mism += mm
            if out:
                rows.append(out)
            if i % 5 == 0 or i == len(jobs):
                print(f"[encounters {i}/{len(jobs)}] found={len(rows)}", flush=True)
    if not rows:
        print("No encounters found.")
        return
    print(f"\nEncounters found: {len(rows)}  (mean perceived distance at branch point {np.mean([r['d0'] for r in rows]):.0f}; "
          f"replay mismatches: {mism} -- must be 0, otherwise branches are not comparable)")
    print("Shaping credit for the tracked agent after the branch. 'alive' = only ticks where the agent survived the step "
          "(pure per-tick credit); 'incl. death' adds the terminal refund. ret = shaped total reward incl. -w_death.")
    print(f"{'branch':<10} {'n':>3} | {'sh alive 5t':>11} {'sh alive 20t':>12} | {'sh incl.death 20t':>17} | {'died<5t%':>8} {'died<20t%':>9} | {'ret 20t':>8}")
    means = {}
    for b in BRANCHES:
        v = lambda key: float(np.mean([r["res"][b][key] for r in rows]))
        means[b] = v("sh_alive_5")
        print(f"{b:<10} {len(rows):>3} | {v('sh_alive_5'):+11.3f} {v('sh_alive_20'):+12.3f} | {v('sh_all_20'):+17.3f} | "
              f"{100*v('died_5'):8.0f} {100*v('died_20'):9.0f} | {v('ret_20'):+8.2f}")
    ok = means["flee"] > means["still"] > means["approach"]
    print(f"\nEarly-credit ordering (sh alive 5t): flee {means['flee']:+.3f} > still {means['still']:+.3f} > "
          f"approach {means['approach']:+.3f}  ->  {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage2_single_predator_evasion")
    ap.add_argument("--episodes", type=int, default=60, help="episodes per policy")
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed-base", type=int, default=777000)
    ap.add_argument("--encounters", type=int, default=40, help="seeds to try for the encounter branching test (0 = skip)")
    ap.add_argument("--w-danger", type=float, default=None, help="override the stage's w_danger")
    ap.add_argument("--danger-radius", type=float, default=None, help="override reward.DANGER_RADIUS")
    ap.add_argument("--phi-mode", choices=("perceived", "true"), default="perceived",
                    help="Phi from the agent's own observation (default) or from the true nearest awake predator")
    args = ap.parse_args()

    stage = get_stage(args.stage)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    bad = [p for p in policies if p not in POLICIES]
    if bad:
        sys.exit(f"unknown policies: {bad}; choose from {POLICIES}")

    n_chunks = math.ceil(args.episodes / CHUNK)
    jobs = [(p, args.seed_base + c, CHUNK, args.stage, args.w_danger, args.danger_radius, args.phi_mode)
            for p in policies for c in range(n_chunks)]
    radius = args.danger_radius if args.danger_radius is not None else R.DANGER_RADIUS
    print(f"stage={args.stage}  policies={policies}  {len(jobs)} jobs  workers={args.workers}  "
          f"DANGER_RADIUS={radius}  PHI_MODE={args.phi_mode}  "
          f"w_danger={'stage default' if args.w_danger is None else args.w_danger}", flush=True)

    res = defaultdict(list)
    w_death = w_danger = None
    t0 = time.time()
    ctx = mp.get_context("spawn") if sys.platform == "win32" else mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for i, (p, eps, w_death, w_danger) in enumerate(pool.imap_unordered(_run_chunk, jobs), 1):
            res[p].extend(eps)
            if i % 5 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"[{i}/{len(jobs)}] elapsed={el/60:.1f}m eta={el/i*(len(jobs)-i)/60:.1f}m", flush=True)

    if not w_danger:
        print("\nWARNING: w_danger is 0 for this stage/override, so shaping columns will be all zeros. "
              "Use a predator stage or pass --w-danger 1.0.")

    print(f"\n=== Reward sanity, {args.stage}, {args.episodes} eps/policy, w_danger={w_danger}, "
          f"DANGER_RADIUS={radius}, PHI_MODE={args.phi_mode}, w_death={w_death} ===")
    hdr = (f"{'policy':<15} {'surv%':>5} {'mean_t':>6} | {'perc.tick%':>10} {'sh/perc.tick':>12} {'|sh| mean':>9} "
           f"{'|sh| p95':>8} {'|sh| max':>8} | {'|base|/tick':>11} {'ratio':>6} | {'popIn/ep':>8} {'popIn|dPhi|':>11} "
           f"{'popOut/ep':>9} | {'sh ret/agent':>12} {'base ret/agent':>14}")
    print(hdr)
    per_perc_mean = {}
    death_rows = []
    for p in policies:
        eps = res[p]
        n = len(eps)
        surv = 100 * sum(1 for e in eps if e["alive"] > 0) / n
        mean_t = np.mean([e["t"] for e in eps])
        ta = sum(e["ticks_alive"] for e in eps)
        tp = sum(e["ticks_perceived"] for e in eps)
        sh = np.concatenate([e["sh_perceived"] for e in eps]) if tp else np.zeros(0, dtype=np.float32)
        abs_base = sum(e["abs_base_alive"] for e in eps) / max(1, ta)
        sh_mean = float(sh.mean()) if sh.size else 0.0
        sh_abs = float(np.abs(sh).mean()) if sh.size else 0.0
        sh_p95 = float(np.percentile(np.abs(sh), 95)) if sh.size else 0.0
        sh_max = float(np.abs(sh).max()) if sh.size else 0.0
        ratio = sh_abs / abs_base if abs_base > 0 else float("nan")
        pin = np.mean([e["pop_in"] for e in eps])
        pout = np.mean([e["pop_out"] for e in eps])
        mags = [m for e in eps for m in e["pop_in_mag"]]
        starts = sum(e["start"] for e in eps)
        sh_ret = sum(e["sh_total"] for e in eps) / max(1, starts)
        base_ret = sum(e["base_total"] for e in eps) / max(1, starts)
        per_perc_mean[p] = sh_mean
        print(f"{p:<15} {surv:5.1f} {mean_t:6.0f} | {100*tp/max(1,ta):10.2f} {sh_mean:+12.4f} {sh_abs:9.4f} "
              f"{sh_p95:8.4f} {sh_max:8.4f} | {abs_base:11.5f} {ratio:6.1f} | {pin:8.2f} "
              f"{(np.mean(mags) if mags else 0.0):11.2f} {pout:9.2f} | {sh_ret:+12.3f} {base_ret:+14.3f}")
        d = sum(e["deaths"] for e in eps)
        dp = sum(e["deaths_perceived"] for e in eps)
        ref = [r for e in eps for r in e["refunds"]]
        death_rows.append((p, d, dp, float(np.mean(ref)) if ref else 0.0))

    print("\nDeath-step refund (Phi(terminal)=0 convention refunds w_danger*|Phi(s_prev)| at death):")
    print(f"{'policy':<15} {'deaths':>7} {'w/ predator perceived':>22} {'mean refund |Phi|':>18} {'effective death penalty':>24}")
    for p, d, dp, mref in death_rows:
        eff = (w_death - (w_danger or 0.0) * mref) if d else float("nan")
        print(f"{p:<15} {d:7d} {dp:22d} {mref:18.3f} {eff:24.3f}")

    print("\nColumn guide:")
    print("  sh/perc.tick   mean shaping reward in ticks where a predator is perceived (before or after the step);")
    print("                 NOT an ordering metric (see the encounter test below for that)")
    print("  ratio          mean|shaping| on perceived ticks / mean|non-death base reward| per tick (how loud shaping is)")
    print("  popIn/popOut   ticks where Phi jumps by more than 0.3 in ONE tick because a predator entered/left perception")
    print("  ret/agent      total reward summed over the episode per starting agent, split into shaping / everything else")
    if args.encounters > 0 and w_danger:
        _run_encounter_test(args, w_danger, radius)


if __name__ == "__main__":
    main()
