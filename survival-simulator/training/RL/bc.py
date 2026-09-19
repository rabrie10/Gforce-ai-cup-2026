"""
Behavior cloning (+ DAgger) of the rule-based heuristic into the RL ActorCritic,
so PPO fine-tuning starts from a policy that already survives instead of from noise.

Why: on stage 1 the heuristic (with the blocked-spawn energy charge removed) survives
~99.5% of episodes, while PPO from scratch plateaus around 71%. Cloning the heuristic
from its own observations is a supervised problem; PPO then only has to improve on it.

Pipeline
  round 0 : run the heuristic itself, record (obs, heuristic action, discounted return).
  round k : run the CURRENT student (deterministic mean action), but label every visited
            state with the heuristic's action (DAgger: fixes covariate shift, the student
            sees the states its own mistakes lead to). Aggregate all data, retrain.
  student : actor mean <- MSE to heuristic action (spawn dim always -1: reproduction is
            blocked/unneeded in stages 1-3; heuristic spawn requests are dropped).
            critic <- MC discounted return (variance-normalised weight so it does not
            dominate the shared trunk). log_std set to --init-log-std afterwards
            (PPO's exploration noise).

Everything is monkeypatch-free: uses CurriculumEnv exactly as train.py/eval do.

Usage (repo root, venv active):
    python3 -u training/RL/bc.py --stage stage1_foraging --workers 14 \
        --rounds 4 --episodes-per-round 80 --out training/RL/checkpoints/stage1_foraging_bc.pt
Then evaluate:
    python3 -u training/RL/eval_checkpoints.py --stage stage1_foraging --iters "" \
        --files training/RL/checkpoints/stage1_foraging_bc.pt --episodes 100 --refine-top 0 --workers 14
and fine-tune with PPO:
    python3 -u training/RL/train.py --stage stage1_foraging --resume-from training/RL/checkpoints/stage1_foraging_bc.pt \
        --lr 1e-4 --iterations 300 --workers 14 --ignore-goal --out training/RL/checkpoints/ft1
"""
import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.controllers.heuristic_policy import action_decision as heuristic_action
from training.RL.curriculum import get_stage
from training.RL.env_wrapper import CurriculumEnv, OBS_DIM
from training.RL.network import ACTION_DIM, ActorCritic

GAMMA = 0.99
CHUNK = 5


def _to_norm(req, sprint_speed):
    """Inverse of env_wrapper.decode_action; spawn dim fixed to -1 (never spawn)."""
    a0 = 2.0 * (req.move_distance / max(sprint_speed, 1e-6)) - 1.0
    a1 = req.move_direction / math.pi
    a2 = req.turn_angle / math.pi
    return np.clip(np.array([a0, a1, a2, -1.0], dtype=np.float32), -1.0, 1.0)


def _returns(rewards):
    out = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + GAMMA * g
        out[t] = g
    return out


def _collect(job):
    """Run n_eps episodes. student_state=None -> heuristic acts; else student (deterministic) acts.
    Every visited state is labelled with the heuristic's action."""
    student_state, seed, n_eps, stage_name = job
    torch.set_num_threads(1)
    stage = get_stage(stage_name)
    env = CurriculumEnv(stage, seed=seed)
    rng = random.Random(seed)
    net = None
    if student_state is not None:
        net = ActorCritic(OBS_DIM, ACTION_DIM)
        net.load_state_dict(student_state)
        net.eval()
    O, A, R = [], [], []
    surv = 0
    for _ in range(n_eps):
        obs = env.reset()
        traj = {}  # aid -> (obs list, act list, rew list)
        while True:
            ids = sorted(obs.keys())
            sim_env = env.sim.env
            expert = {}
            for aid in ids:
                ag = sim_env.agents_dict.get(aid)
                req = heuristic_action(sim_env.get_agent_state(aid), rng)
                expert[aid] = _to_norm(req, ag.sprint_speed)
            if net is None:
                acts = expert
            else:
                batch = torch.as_tensor(np.stack([obs[a] for a in ids]), dtype=torch.float32)
                with torch.no_grad():
                    mean, _, _ = net.forward(batch)
                m = mean.numpy()
                acts = {aid: m[i] for i, aid in enumerate(ids)}
            for aid in ids:
                o, a, r = traj.setdefault(aid, ([], [], []))
                o.append(obs[aid])
                a.append(expert[aid])
            obs, rewards, done, info = env.step(acts)
            for aid, r in rewards.items():
                if aid in traj:
                    traj[aid][2].append(r)
            if done:
                surv += int(info["num_agents"] > 0)
                break
        for aid, (o, a, r) in traj.items():
            n = min(len(o), len(r))
            if n == 0:
                continue
            O.append(np.stack(o[:n]))
            A.append(np.stack(a[:n]))
            R.append(_returns(r[:n]))
    return np.concatenate(O), np.concatenate(A), np.concatenate(R), surv, n_eps


def collect_round(pool, student_state, n_eps, seed_base, stage_name):
    n_chunks = math.ceil(n_eps / CHUNK)
    jobs = [(student_state, seed_base + c, CHUNK, stage_name) for c in range(n_chunks)]
    O, A, R, S, N = [], [], [], 0, 0
    for o, a, r, s, n in pool.imap_unordered(_collect, jobs):
        O.append(o); A.append(a); R.append(r); S += s; N += n
    return np.concatenate(O), np.concatenate(A), np.concatenate(R), S / max(N, 1)


def fit(net, O, A, R, epochs, lr, bs=512):
    obs = torch.as_tensor(O, dtype=torch.float32)
    act = torch.as_tensor(A, dtype=torch.float32)
    ret = torch.as_tensor(R, dtype=torch.float32)
    r_mean, r_std = ret.mean(), ret.std() + 1e-6
    ret_n = (ret - r_mean) / r_std  # critic head fit on normalised returns, un-normalised below
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = obs.shape[0]
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot_a = tot_v = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            mean, _, value = net.forward(obs[idx])
            loss_a = F.mse_loss(mean, act[idx])
            loss_v = F.smooth_l1_loss(value, ret_n[idx])
            loss = loss_a + 0.5 * loss_v
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot_a += float(loss_a) * len(idx); tot_v += float(loss_v) * len(idx)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:3d}  actor_mse={tot_a/n:.4f}  value_huber(norm)={tot_v/n:.4f}", flush=True)
    net.eval()
    return float(r_mean), float(r_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage1_foraging")
    ap.add_argument("--rounds", type=int, default=4, help="round 0 = heuristic rollouts, rounds 1.. = DAgger")
    ap.add_argument("--episodes-per-round", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-log-std", type=float, default=-1.5, help="PPO exploration std afterwards (-1.5 -> 0.22)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed-base", type=int, default=5_000_000, help="kept far from eval seeds (777000/877000)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "checkpoints", "stage1_foraging_bc.pt"))
    args = ap.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    net = ActorCritic(OBS_DIM, ACTION_DIM)
    ctx = mp.get_context("spawn") if sys.platform == "win32" else mp.get_context("fork")
    Os, As, Rs = [], [], []
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        for rd in range(args.rounds):
            state = None if rd == 0 else {k: v.clone() for k, v in net.state_dict().items()}
            O, A, R, surv = collect_round(pool, state, args.episodes_per_round,
                                          args.seed_base + rd * 1000, args.stage)
            who = "heuristic" if rd == 0 else "student(det)"
            print(f"[round {rd}] acting policy={who}  survive={surv*100:.1f}%  new samples={len(O)}  "
                  f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
            Os.append(O); As.append(A); Rs.append(R)
            r_mean, r_std = fit(net, np.concatenate(Os), np.concatenate(As), np.concatenate(Rs),
                args.epochs if rd < args.rounds - 1 else args.epochs * 2, args.lr)
    with torch.no_grad():
        # Undo the return normalisation inside the critic head: values in real reward units.
        net.critic.weight.mul_(r_std)
        net.critic.bias.mul_(r_std).add_(r_mean)
        net.log_std.fill_(args.init_log_std)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
