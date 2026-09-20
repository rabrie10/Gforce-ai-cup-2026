"""
Optuna hyper-parameter search around train.train(), starting every trial from the same
BC checkpoint so trials differ only in the PPO settings.

    pip install optuna
    python3 -u training/RL/tune.py --stage stage3_realistic_predator_pressure \
        --resume-from training/RL/checkpoints/stage5_bc.pt --init-log-std -3.5 \
        --trial-iters 40 --n-trials 12 --workers 14 --central-critic \
        --out training/RL/checkpoints/optuna_s3

Objective = mean of the last 3 rolling-metric values of the trial (survive rate on
stages that define it, else mean colony survival time in seconds). It is a noisy
training-time metric, so the search only narrows the settings; the winner must still be
confirmed with eval_baselines / eval_checkpoints paired against the heuristic.

The study is stored in SQLite (<out>/study.db): re-running the same command resumes it.
"""
import argparse
import csv
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import optuna

from training.RL import train as T


def _score(iters_csv: str) -> float:
    vals = []
    with open(iters_csv) as f:
        for row in csv.DictReader(f):
            if row["roll_metric"] not in ("", None):
                vals.append(float(row["roll_metric"]))
    if not vals:
        return float("-inf")
    tail = vals[-3:]
    return sum(tail) / len(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--resume-from", required=True)
    ap.add_argument("--init-log-std", type=float, default=-3.5)
    ap.add_argument("--trial-iters", type=int, default=40)
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--central-critic", action="store_true")
    ap.add_argument("--critic-warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            lr=trial.suggest_float("lr", 2e-5, 3e-4, log=True),
            ent_coef=trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True),
            clip_eps=trial.suggest_categorical("clip_eps", [0.1, 0.15, 0.2, 0.3]),
            gae_lambda=trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
            colony_bonus=trial.suggest_float("colony_bonus", 0.0, 0.2),
            rollout_steps=trial.suggest_categorical("rollout_steps", [256, 512, 1024]),
        )
        tdir = os.path.join(args.out, f"trial{trial.number}")
        T.train(
            stage_name=args.stage, iterations=args.trial_iters, rollout_steps=params.pop("rollout_steps"),
            seed=1000 + trial.number, out_dir=tdir, log_every=10, checkpoint_every=10**9,
            resume_from=args.resume_from, workers=args.workers, ignore_goal=True,
            init_log_std=args.init_log_std, central_critic=args.central_critic,
            critic_warmup=args.critic_warmup if args.central_critic else 0, **params,
        )
        return _score(os.path.join(tdir, f"{args.stage}_iters.csv"))

    study = optuna.create_study(
        direction="maximize", study_name="ppo", storage=f"sqlite:///{os.path.join(args.out, 'study.db')}",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=args.n_trials)
    print("\nBEST", study.best_value)
    for k, v in study.best_params.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
