"""
Multi-seed sweep of the rule-based heuristic policy, headless.

Each seed is a fully independent episode -- its own RNG, its own Environment,
no shared state -- so seeds are run in parallel across processes by default.

Usage:
    python3 training/rule_based/sweep.py --seeds 1-30 --max-sim-time 3000
    python3 training/rule_based/sweep.py --seeds 1 2 3 7 11 --max-sim-time 600 --workers 4
"""
import argparse
import multiprocessing as mp
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pandas as pd

from training.rule_based.run_episode import run_episode, save_run

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _parse_seeds(raw):
    """Accepts '1-30', '1,2,3', or a mix like '1-5 10 20-22'."""
    seeds = []
    for chunk in raw:
        for part in str(chunk).replace(",", " ").split():
            if "-" in part:
                lo, hi = part.split("-")
                seeds.extend(range(int(lo), int(hi) + 1))
            else:
                seeds.append(int(part))
    # De-dupe, keep order
    seen = set()
    out = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _run_one(job):
    seed, max_sim_time = job
    result = run_episode(seed=seed, max_sim_time=max_sim_time)
    out_dir = os.path.join(RESULTS_DIR, f"run_seed{seed}")
    save_run(result, out_dir)
    return result["summary"]


def _default_worker_count() -> int:
    """
    Conservative on purpose. Each worker briefly does real CPU + memory work
    building a fresh Environment (obstacles, the biome map, spatial grids,
    several pygame Surfaces). cpu_count()-1 workers hitting that at once is
    what made an earlier version of this script bog down the machine it ran
    on -- see patch_environment_for_headless() in instrumentation.py for the
    biggest single fix, but a smaller worker pool is still safer as a default.
    Override with --workers once you've confirmed your machine handles it fine.
    """
    return min(4, max(1, mp.cpu_count() - 1))


def run_sweep(seeds, max_sim_time: float = 3000.0, n_workers: int = None) -> pd.DataFrame:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    jobs = [(s, max_sim_time) for s in seeds]

    n_workers = n_workers or _default_worker_count()
    n_workers = min(n_workers, len(jobs))

    summaries = []
    if n_workers <= 1:
        for job in jobs:
            summary = _run_one(job)
            summaries.append(summary)
            print(f"seed {summary['seed']:>4}  extinction_time={summary['extinction_time']}  "
                  f"score={summary['final_score']:.2f}  deaths(pred/old/young)="
                  f"{summary['deaths_predator']}/{summary['deaths_starvation_old']}/{summary['deaths_starvation_young']}")
    else:
        with mp.Pool(n_workers) as pool:
            for summary in pool.imap_unordered(_run_one, jobs):
                summaries.append(summary)
                print(f"seed {summary['seed']:>4}  extinction_time={summary['extinction_time']}  "
                      f"score={summary['final_score']:.2f}  deaths(pred/old/young)="
                      f"{summary['deaths_predator']}/{summary['deaths_starvation_old']}/{summary['deaths_starvation_young']}")

    df = pd.DataFrame(summaries).sort_values("seed").reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS_DIR, "sweep_summary.csv"), index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Sweep the rule-based heuristic across many seeds.")
    parser.add_argument("--seeds", type=str, nargs="+", default=["1-30"], help="e.g. 1-30 or 1 2 3 7-9")
    parser.add_argument("--max-sim-time", type=float, default=3000.0)
    parser.add_argument("--workers", type=int, default=None,
                         help=f"Parallel worker processes. Default is conservative "
                              f"(min(4, cpu_count()-1) = {_default_worker_count()} on this machine) "
                              f"since each worker briefly does real CPU+memory work per new environment. "
                              f"Raise it once you've confirmed your machine handles that fine.")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    workers = args.workers or _default_worker_count()
    print(f"Running {len(seeds)} seed(s), max_sim_time={args.max_sim_time}, workers={workers}")
    df = run_sweep(seeds, max_sim_time=args.max_sim_time, n_workers=args.workers)
    print("\nSummary across seeds:")
    print(df[["extinction_time", "final_score", "deaths_predator", "deaths_starvation_old", "deaths_starvation_young"]].describe())
    print(f"\nSaved per-seed data to {RESULTS_DIR}/run_seed<seed>/ and {RESULTS_DIR}/sweep_summary.csv")


if __name__ == "__main__":
    main()
