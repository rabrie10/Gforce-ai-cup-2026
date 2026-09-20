"""
Evaluate saved checkpoints of one curriculum stage on FRESH episodes and rank
them, so you can pick the best one instead of trusting `*_final.pt`.

- Same env/stage config as training (CurriculumEnv + get_stage), same
  goal metric as curriculum.py's stage 1 ("survive_rate": >=1 agent alive at
  the stage's max_sim_time).
- Every checkpoint sees the SAME episode seeds (paired comparison), so
  differences between checkpoints are not just seed luck.
- Two policy modes per checkpoint:
    stoch = sampled actions (how training/goal-check ran; what stage 2 will
            start from)
    det   = tanh-mean action (what inference.py ships)
- --mask-spawn simulates the fix for the "blocked reproduction still costs 100 energy" issue
  (stages 1-3); "wasted spawn/ep" counts spawn requests that cost (or would have cost) 100
  energy with no child created. See spawn_hook.py.
- Two passes: pass 1 = cheap screen of all candidates; pass 2 = extra
  episodes for the top-K only (ranked by mean survive-rate over both modes),
  merged with pass 1.

Usage (from repo root, venv active):
    python3 -u training/RL/eval_checkpoints.py --stage stage1_foraging \
        --iters 400-520,1180-1340 --include-final --workers 14
"""
import argparse
import csv
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import defaultdict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from training.RL.curriculum import get_stage
from training.RL.env_wrapper import CurriculumEnv, OBS_DIM
from training.RL.network import ACTION_DIM, ActorCritic
from training.RL.spawn_hook import hook_spawn

CHUNK = 10  # episodes per job (amortizes env construction)
MODES = ("stoch", "det")


def _run_chunk(job):
    ckpt, mode, seed, n_eps, stage_name, mask_spawn = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    stage = get_stage(stage_name)
    env = CurriculumEnv(stage, seed=seed)
    net = ActorCritic(OBS_DIM, ACTION_DIM)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    det = mode == "det"
    out = []
    for _ in range(n_eps):
        obs = env.reset()
        spawn_counts = {"wasted": 0}
        hook_spawn(env.sim.env, spawn_counts, mask_spawn)
        info = {"sim_time": 0.0, "num_agents": 0}
        while obs:
            ids = sorted(obs.keys())
            batch = torch.as_tensor(np.stack([obs[a] for a in ids]), dtype=torch.float32)
            act, _, _, _ = net.act(batch, deterministic=det)
            act_np = act.numpy()
            obs, _, done, info = env.step({aid: act_np[i] for i, aid in enumerate(ids)})
            if done:
                break
        out.append((ckpt, mode, float(info["sim_time"]), int(info["num_agents"]), spawn_counts["wasted"]))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def pick_checkpoints(ckpt_dir, stage, ranges, include_final, files=None):
    pat = re.compile(rf"^{re.escape(stage)}_iter(\d+)\.pt$")
    found = {}
    for f in os.listdir(ckpt_dir):
        m = pat.match(f)
        if m:
            found[int(m.group(1))] = os.path.join(ckpt_dir, f)
    chosen = {}
    for r in ranges.split(","):
        r = r.strip()
        if not r:
            continue
        lo, hi = (int(x) for x in r.split("-")) if "-" in r else (int(r), int(r))
        for it, p in found.items():
            if lo <= it <= hi:
                chosen[p] = f"iter{it}"
    for fp in (files or []):
        chosen[os.path.abspath(fp)] = os.path.basename(fp).replace('.pt', '')
    if include_final:
        fp = os.path.join(ckpt_dir, f"{stage}_final.pt")
        if os.path.exists(fp):
            chosen[fp] = "final"
    return chosen


def run_jobs(jobs, workers, label):
    results = defaultdict(list)
    t0 = time.time()
    done_jobs = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for chunk in pool.imap_unordered(_run_chunk, jobs):
            for ckpt, mode, t, n_agents, wasted in chunk:
                results[(ckpt, mode)].append((t, n_agents, wasted))
            done_jobs += 1
            if done_jobs % 20 == 0 or done_jobs == len(jobs):
                el = time.time() - t0
                eta = el / done_jobs * (len(jobs) - done_jobs)
                print(f"[{label}] {done_jobs}/{len(jobs)} jobs  elapsed={el/60:.1f}m  eta={eta/60:.1f}m", flush=True)
    return results


def make_jobs(ckpts, n_eps, seed_base, stage, mask_spawn=False):
    n_chunks = math.ceil(n_eps / CHUNK)
    jobs = []
    for ckpt in ckpts:
        for mode in MODES:
            for c in range(n_chunks):
                jobs.append((ckpt, mode, seed_base + c, CHUNK, stage, mask_spawn))
    return jobs


def summarize(results, names):
    rows = []
    by_ckpt = defaultdict(dict)
    for (ckpt, mode), eps in results.items():
        n = len(eps)
        k = sum(1 for e in eps if e[1] > 0)
        lo, hi = wilson(k, n)
        by_ckpt[ckpt][mode] = dict(
            n=n, rate=k / n, lo=lo, hi=hi,
            mean_t=float(np.mean([e[0] for e in eps])),
            mean_alive=float(np.mean([e[1] for e in eps])),
            wasted=float(np.mean([e[2] for e in eps])),
        )
    for ckpt, d in by_ckpt.items():
        score = float(np.mean([d[m]["rate"] for m in MODES if m in d]))
        rows.append((score, names[ckpt], ckpt, d))
    rows.sort(key=lambda r: -r[0])
    return rows


def print_table(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'checkpoint':<10} {'avg':>5} | {'stoch surv%':>11} {'95% CI':>13} {'mean_t':>7} | {'det surv%':>9} {'95% CI':>13} {'mean_t':>7} | {'wasted spawn/ep (stoch, det)':>28} | n(each)")
    for score, name, _, d in rows:
        blank = dict(n=0, rate=float('nan'), lo=float('nan'), hi=float('nan'), mean_t=float('nan'), mean_alive=float('nan'), wasted=float('nan'))
        s, t = d.get("stoch", blank), d.get("det", blank)
        n_ = s['n'] or t['n']
        print(f"{name:<10} {score*100:5.1f} | {s['rate']*100:11.1f} [{s['lo']*100:4.0f},{s['hi']*100:4.0f}]   {s['mean_t']:7.1f} | "
              f"{t['rate']*100:9.1f} [{t['lo']*100:4.0f},{t['hi']*100:4.0f}]   {t['mean_t']:7.1f} | {s['wasted']:12.2f} {t['wasted']:>15.2f} | {n_}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage1_foraging")
    ap.add_argument("--ckpt-dir", default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    ap.add_argument("--iters", default="400-520,1180-1340", help="comma list of iteration ranges, e.g. 400-520,1180-1340")
    ap.add_argument("--include-final", action="store_true")
    ap.add_argument("--modes", default="stoch,det", help="comma list of stoch,det (e.g. --modes stoch to skip deterministic)")
    ap.add_argument("--files", default="", help="comma list of extra checkpoint .pt paths (e.g. bc / best files)")
    ap.add_argument("--episodes", type=int, default=100, help="pass-1 episodes per (checkpoint, mode)")
    ap.add_argument("--refine-top", type=int, default=4)
    ap.add_argument("--refine-episodes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed-base", type=int, default=777000)
    ap.add_argument("--mask-spawn", action="store_true",
                    help="Simulate the FIX for the blocked-reproduction energy charge (force spawn_agent=False; "
                         "stages 1-3 only). Default: current behaviour (blocked spawn requests still cost 100 energy).")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "eval_results"))
    args = ap.parse_args()

    if args.mask_spawn and get_stage(args.stage).reproduction:
        sys.exit(f"--mask-spawn is only meaningful for stages with reproduction disabled; {args.stage} has it enabled.")
    print(f"spawn charge: {'MASKED (fixed behaviour)' if args.mask_spawn else 'ACTIVE (current behaviour)'}", flush=True)
    global MODES
    MODES = tuple(m for m in args.modes.split(",") if m in ("stoch", "det"))
    names = pick_checkpoints(args.ckpt_dir, args.stage, args.iters, args.include_final, [f for f in args.files.split(',') if f])
    if not names:
        sys.exit("No checkpoints matched.")
    ckpts = sorted(names)
    print(f"{len(ckpts)} checkpoints, {args.workers} workers, stage={args.stage}", flush=True)
    for c in ckpts:
        print("  ", names[c], flush=True)

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    res1 = run_jobs(make_jobs(ckpts, args.episodes, args.seed_base, args.stage, args.mask_spawn), args.workers, "pass1")
    rows1 = summarize(res1, names)
    print_table(rows1, f"PASS 1 ({args.episodes} eps per checkpoint per mode)")

    merged = defaultdict(list)
    for k, v in res1.items():
        merged[k].extend(v)

    if args.refine_top > 0 and args.refine_episodes > 0:
        top = [r[2] for r in rows1[: args.refine_top]]
        print(f"\nRefining top {len(top)}: {[names[c] for c in top]}", flush=True)
        res2 = run_jobs(make_jobs(top, args.refine_episodes, args.seed_base + 100000, args.stage, args.mask_spawn), args.workers, "pass2")
        for k, v in res2.items():
            merged[k].extend(v)
        rows2 = summarize({k: v for k, v in merged.items() if k[0] in top}, names)
        print_table(rows2, f"PASS 1+2 MERGED (top {len(top)}, {args.episodes + args.refine_episodes} eps each per mode)")
        best = rows2[0][1]
        print(f"\nBest by avg survive-rate: {best}  ({rows2[0][2]})")

    raw_path = os.path.join(args.out, f"{args.stage}_eval_raw_{stamp}.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "mode", "sim_time", "final_num_agents", "wasted_spawn"])
        for (ckpt, mode), eps in merged.items():
            for t, a, wsp in eps:
                w.writerow([names[ckpt], mode, t, a, wsp])
    print(f"\nRaw episodes: {raw_path}")


if __name__ == "__main__":
    main()
