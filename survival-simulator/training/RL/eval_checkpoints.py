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

CHUNK = 10  # episodes per job (amortizes env construction)
MODES = ("stoch", "det")


def _run_chunk(job):
    ckpt, mode, seed, n_eps, stage_name = job
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
        info = {"sim_time": 0.0, "num_agents": 0}
        while obs:
            ids = sorted(obs.keys())
            batch = torch.as_tensor(np.stack([obs[a] for a in ids]), dtype=torch.float32)
            act, _, _, _ = net.act(batch, deterministic=det)
            act_np = act.numpy()
            obs, _, done, info = env.step({aid: act_np[i] for i, aid in enumerate(ids)})
            if done:
                break
        out.append((ckpt, mode, float(info["sim_time"]), int(info["num_agents"])))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def pick_checkpoints(ckpt_dir, stage, ranges, include_final):
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
            for ckpt, mode, t, n_agents in chunk:
                results[(ckpt, mode)].append((t, n_agents))
            done_jobs += 1
            if done_jobs % 20 == 0 or done_jobs == len(jobs):
                el = time.time() - t0
                eta = el / done_jobs * (len(jobs) - done_jobs)
                print(f"[{label}] {done_jobs}/{len(jobs)} jobs  elapsed={el/60:.1f}m  eta={eta/60:.1f}m", flush=True)
    return results


def make_jobs(ckpts, n_eps, seed_base, stage):
    n_chunks = math.ceil(n_eps / CHUNK)
    jobs = []
    for ckpt in ckpts:
        for mode in MODES:
            for c in range(n_chunks):
                jobs.append((ckpt, mode, seed_base + c, CHUNK, stage))
    return jobs


def summarize(results, names):
    rows = []
    by_ckpt = defaultdict(dict)
    for (ckpt, mode), eps in results.items():
        n = len(eps)
        k = sum(1 for _, a in eps if a > 0)
        lo, hi = wilson(k, n)
        by_ckpt[ckpt][mode] = dict(
            n=n, rate=k / n, lo=lo, hi=hi,
            mean_t=float(np.mean([t for t, _ in eps])),
            mean_alive=float(np.mean([a for _, a in eps])),
        )
    for ckpt, d in by_ckpt.items():
        score = float(np.mean([d[m]["rate"] for m in MODES if m in d]))
        rows.append((score, names[ckpt], ckpt, d))
    rows.sort(key=lambda r: -r[0])
    return rows


def print_table(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'checkpoint':<10} {'avg':>5} | {'stoch surv%':>11} {'95% CI':>13} {'mean_t':>7} | {'det surv%':>9} {'95% CI':>13} {'mean_t':>7} | n(each)")
    for score, name, _, d in rows:
        s, t = d["stoch"], d["det"]
        print(f"{name:<10} {score*100:5.1f} | {s['rate']*100:11.1f} [{s['lo']*100:4.0f},{s['hi']*100:4.0f}]   {s['mean_t']:7.1f} | "
              f"{t['rate']*100:9.1f} [{t['lo']*100:4.0f},{t['hi']*100:4.0f}]   {t['mean_t']:7.1f} | {s['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage1_foraging")
    ap.add_argument("--ckpt-dir", default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    ap.add_argument("--iters", default="400-520,1180-1340", help="comma list of iteration ranges, e.g. 400-520,1180-1340")
    ap.add_argument("--include-final", action="store_true")
    ap.add_argument("--episodes", type=int, default=100, help="pass-1 episodes per (checkpoint, mode)")
    ap.add_argument("--refine-top", type=int, default=4)
    ap.add_argument("--refine-episodes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed-base", type=int, default=777000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "eval_results"))
    args = ap.parse_args()

    names = pick_checkpoints(args.ckpt_dir, args.stage, args.iters, args.include_final)
    if not names:
        sys.exit("No checkpoints matched.")
    ckpts = sorted(names)
    print(f"{len(ckpts)} checkpoints, {args.workers} workers, stage={args.stage}", flush=True)
    for c in ckpts:
        print("  ", names[c], flush=True)

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    res1 = run_jobs(make_jobs(ckpts, args.episodes, args.seed_base, args.stage), args.workers, "pass1")
    rows1 = summarize(res1, names)
    print_table(rows1, f"PASS 1 ({args.episodes} eps per checkpoint per mode)")

    merged = defaultdict(list)
    for k, v in res1.items():
        merged[k].extend(v)

    if args.refine_top > 0 and args.refine_episodes > 0:
        top = [r[2] for r in rows1[: args.refine_top]]
        print(f"\nRefining top {len(top)}: {[names[c] for c in top]}", flush=True)
        res2 = run_jobs(make_jobs(top, args.refine_episodes, args.seed_base + 100000, args.stage), args.workers, "pass2")
        for k, v in res2.items():
            merged[k].extend(v)
        rows2 = summarize({k: v for k, v in merged.items() if k[0] in top}, names)
        print_table(rows2, f"PASS 1+2 MERGED (top {len(top)}, {args.episodes + args.refine_episodes} eps each per mode)")
        best = rows2[0][1]
        print(f"\nBest by avg survive-rate: {best}  ({rows2[0][2]})")

    raw_path = os.path.join(args.out, f"{args.stage}_eval_raw_{stamp}.csv")
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "mode", "sim_time", "final_num_agents"])
        for (ckpt, mode), eps in merged.items():
            for t, a in eps:
                w.writerow([names[ckpt], mode, t, a])
    print(f"\nRaw episodes: {raw_path}")


if __name__ == "__main__":
    main()
