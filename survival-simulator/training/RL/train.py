"""
IPPO (independent PPO, shared weights) trainer for one curriculum stage.
Runs entirely on obs/action encodings from env_wrapper.py -- never touches
src/. See network.py's docstring for why IPPO (not MAPPO/attention) is the
v1 target.

Per-agent-id trajectories are tracked across ticks (agent_id is a strictly
increasing counter in src/elements/environment.py, never reused, so keying
on it is safe). A trajectory is finalized -- and handed to GAE -- either
when the agent dies (true terminal, bootstrap value = 0) or when the
rollout window / max_sim_time cuts it off while still alive (truncation,
bootstrap value = the critic's current estimate, NOT zero -- otherwise the
value function would be taught that hitting the time limit is as bad as
dying, which it isn't).

Multi-worker note: this workload is CPU-bound Python-object-simulation
(env.step) and tiny-batch network calls (3-50 agents at a time), not
matrix-math-bound -- profiled at ~1-4ms/tick, almost entirely Python/
framework call overhead. A GPU would sit idle waiting on batches this
small. What actually helps is running several independent environments in
parallel across CPU cores (--workers > 1): each worker owns a persistent
env + its own copy of the current weights, collects a rollout, and sends
finished trajectories back for one pooled PPO update per iteration. Same
conservative-default-worker-count lesson as ../rule_based/sweep.py (don't
default to cpu_count() and overwhelm the machine).

Usage:
    python3 training/RL/train.py --stage stage1_foraging --iterations 200 --workers 4
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time as wallclock
from dataclasses import dataclass, field
from typing import Dict, List, Optional

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn

from training.RL.curriculum import get_stage
from training.RL.env_wrapper import CurriculumEnv, OBS_DIM
from training.RL.network import ACTION_DIM, ActorCritic

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
LR = 3e-4
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256


@dataclass
class _Traj:
    obs: List[np.ndarray] = field(default_factory=list)
    action: List[np.ndarray] = field(default_factory=list)
    logprob: List[float] = field(default_factory=list)
    value: List[float] = field(default_factory=list)
    reward: List[float] = field(default_factory=list)
    bootstrap: float = 0.0  # value estimate to bootstrap from after the LAST recorded step


def _gae(traj: _Traj, gamma: float, lam: float):
    """Returns (advantages, returns) arrays, same length as traj.reward."""
    T = len(traj.reward)
    values = traj.value + [traj.bootstrap]
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        delta = traj.reward[t] + gamma * values[t + 1] - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(traj.value, dtype=np.float32)
    return advantages, returns


def collect_rollout(env: CurriculumEnv, net: ActorCritic, obs: Dict[int, np.ndarray],
                     rollout_steps: int, device: torch.device, episode_log: List[dict]):
    """
    Runs rollout_steps env ticks (resetting and continuing into new episodes
    as needed), returns (finished trajectories ready for GAE, the obs dict
    to resume from next call). Appends one dict per completed episode to
    episode_log (score, extinction_time/truncated, final_num_agents).
    """
    active: Dict[int, _Traj] = {}
    finished: List[_Traj] = []

    ep_start_time = wallclock.time()

    for _ in range(rollout_steps):
        agent_ids = sorted(obs.keys())
        obs_batch = torch.as_tensor(np.stack([obs[a] for a in agent_ids]), dtype=torch.float32, device=device)
        action_clipped, action_raw, logprob, value = net.act(obs_batch)
        action_clipped_np = action_clipped.cpu().numpy()
        action_raw_np = action_raw.cpu().numpy()
        logprob_np = logprob.cpu().numpy()
        value_np = value.cpu().numpy()

        for i, aid in enumerate(agent_ids):
            traj = active.setdefault(aid, _Traj())
            traj.obs.append(obs[aid])
            traj.action.append(action_raw_np[i])
            traj.logprob.append(float(logprob_np[i]))
            traj.value.append(float(value_np[i]))

        actions_norm = {aid: action_clipped_np[i] for i, aid in enumerate(agent_ids)}
        next_obs, rewards, done, info = env.step(actions_norm)

        for aid, r in rewards.items():
            if aid not in active:
                continue
            active[aid].reward.append(r)
            if aid not in next_obs:
                # true terminal: agent died this tick, nothing to bootstrap from
                active[aid].bootstrap = 0.0
                finished.append(active.pop(aid))

        obs = next_obs

        if done:
            # Episode ended (extinction or max_sim_time). Anyone still active
            # survived to the cutoff -- truncation, not death: bootstrap from
            # the critic's own value estimate of their final observation.
            if active:
                remaining_ids = sorted(active.keys())
                if remaining_ids:
                    rem_batch = torch.as_tensor(
                        np.stack([obs[a] for a in remaining_ids]), dtype=torch.float32, device=device
                    )
                    with torch.no_grad():
                        _, _, rem_value = net.forward(rem_batch)
                    rem_value_np = rem_value.cpu().numpy()
                    for i, aid in enumerate(remaining_ids):
                        active[aid].bootstrap = float(rem_value_np[i])
                        finished.append(active.pop(aid))

            episode_log.append({
                "score": info["score"],
                "sim_time": info["sim_time"],
                "final_num_agents": info["num_agents"],
                "extinct": info["num_agents"] == 0,
                "wall_seconds": wallclock.time() - ep_start_time,
            })
            ep_start_time = wallclock.time()
            obs = env.reset()

    # Rollout window ended with some agents still mid-episode: bootstrap and
    # finalize them too (same truncation logic as above), so no partial
    # trajectory data is silently dropped between calls.
    if active:
        remaining_ids = sorted(active.keys())
        rem_batch = torch.as_tensor(np.stack([obs[a] for a in remaining_ids]), dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, rem_value = net.forward(rem_batch)
        rem_value_np = rem_value.cpu().numpy()
        for i, aid in enumerate(remaining_ids):
            active[aid].bootstrap = float(rem_value_np[i])
            finished.append(active.pop(aid))

    return finished, obs


def _worker_loop(rank: int, stage_name: str, seed: Optional[int], rollout_steps: int,
                  in_q: "mp.Queue", out_q: "mp.Queue") -> None:
    """
    Persistent worker process: builds ONE CurriculumEnv and keeps it (and
    its obs dict) alive across iterations -- rebuilding it per call would
    both lose episode continuity and re-pay the ~biome_map construction
    cost every time. torch.set_num_threads(1) is required here: without
    it, each of N worker processes would spawn its own intra-op thread
    pool sized to the WHOLE machine's core count, oversubscribing by ~N.x
    for a network this tiny that doesn't benefit from multithreading
    anyway.
    """
    torch.set_num_threads(1)
    worker_seed = None if seed is None else seed + rank
    stage = get_stage(stage_name)
    env = CurriculumEnv(stage, seed=worker_seed)
    obs = env.reset()
    net = ActorCritic(OBS_DIM, ACTION_DIM)
    net.eval()

    while True:
        msg = in_q.get()
        if msg is None:
            break
        state_dict = msg
        net.load_state_dict(state_dict)
        episode_log: List[dict] = []
        trajs, obs = collect_rollout(env, net, obs, rollout_steps, torch.device("cpu"), episode_log)
        out_q.put((trajs, episode_log))


def _default_worker_count(requested: Optional[int]) -> int:
    if requested is not None:
        return max(1, requested)
    # Conservative default -- see sweep.py's history: cpu_count() alone
    # once made a whole machine sluggish under parallel load.
    return min(4, max(1, mp.cpu_count() - 1))


def collect_rollout_parallel(workers: List[dict], net: ActorCritic) -> List[_Traj]:
    """
    workers: list of {"in_q": ..., "out_q": ...} for each live worker
    process. Broadcasts the current (CPU) state_dict to every worker,
    blocks until all have replied, and returns the pooled trajectories.
    """
    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    for w in workers:
        w["in_q"].put(state_dict)

    all_trajs: List[_Traj] = []
    all_episode_logs: List[dict] = []
    for w in workers:
        trajs, episode_log = w["out_q"].get()
        all_trajs.extend(trajs)
        all_episode_logs.extend(episode_log)
    return all_trajs, all_episode_logs


def ppo_update(net: ActorCritic, optimizer: torch.optim.Optimizer, trajs: List[_Traj], device: torch.device):
    obs_all, action_all, logprob_all, adv_all, ret_all = [], [], [], [], []
    for traj in trajs:
        if not traj.reward:
            continue
        adv, ret = _gae(traj, GAMMA, GAE_LAMBDA)
        obs_all.append(np.stack(traj.obs))
        action_all.append(np.stack(traj.action))
        logprob_all.append(np.asarray(traj.logprob, dtype=np.float32))
        adv_all.append(adv)
        ret_all.append(ret)

    obs_t = torch.as_tensor(np.concatenate(obs_all), dtype=torch.float32, device=device)
    action_t = torch.as_tensor(np.concatenate(action_all), dtype=torch.float32, device=device)
    old_logprob_t = torch.as_tensor(np.concatenate(logprob_all), dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(np.concatenate(adv_all), dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(np.concatenate(ret_all), dtype=torch.float32, device=device)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = obs_t.shape[0]
    last_policy_loss = last_value_loss = last_entropy = 0.0
    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, MINIBATCH_SIZE):
            idx = perm[start:start + MINIBATCH_SIZE]
            new_logprob, entropy, value = net.evaluate_actions(obs_t[idx], action_t[idx])
            ratio = torch.exp(new_logprob - old_logprob_t[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (value - ret_t[idx]).pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + VF_COEF * value_loss + ENT_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy.mean().item())

    return {"policy_loss": last_policy_loss, "value_loss": last_value_loss, "entropy": last_entropy, "n_transitions": n}


def train(stage_name: str, iterations: int, rollout_steps: int, seed: Optional[int],
          out_dir: str, log_every: int = 5, checkpoint_every: int = 20,
          resume_from: Optional[str] = None, workers: Optional[int] = None) -> str:
    device = torch.device("cpu")
    n_workers = _default_worker_count(workers)
    if n_workers > 1:
        # Leave CPU headroom for the worker processes' own single-threaded
        # torch calls (see _worker_loop) instead of this process's PPO
        # update contending for every core.
        torch.set_num_threads(1)

    net = ActorCritic(OBS_DIM, ACTION_DIM).to(device)
    if resume_from:
        net.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"Resumed weights from {resume_from}")
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{stage_name}_episodes.csv")
    log_f = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_f, fieldnames=["iteration", "score", "sim_time", "final_num_agents", "extinct", "wall_seconds"])
    log_writer.writeheader()

    all_episode_scores: List[float] = []
    t0 = wallclock.time()

    worker_procs: List[dict] = []
    env = None
    obs = None

    if n_workers > 1:
        # "fork" doesn't exist on Windows (no fork() syscall) -- multiprocessing
        # only offers "spawn" there, which re-imports this module fresh in
        # each worker process instead of copying the parent's memory. That's
        # exactly why _worker_loop and everything it needs are plain
        # module-level functions/imports rather than closures: spawn can
        # pickle and re-create them from scratch. On Linux/Mac, fork is used
        # since it's cheaper (no re-import, no re-pickling of args).
        ctx = mp.get_context("spawn") if sys.platform == "win32" else mp.get_context("fork")
        for rank in range(n_workers):
            in_q, out_q = ctx.Queue(), ctx.Queue()
            p = ctx.Process(target=_worker_loop, args=(rank, stage_name, seed, rollout_steps, in_q, out_q), daemon=True)
            p.start()
            worker_procs.append({"process": p, "in_q": in_q, "out_q": out_q})
        print(f"Started {n_workers} parallel rollout workers.")
    else:
        stage = get_stage(stage_name)
        env = CurriculumEnv(stage, seed=seed)
        obs = env.reset()

    try:
        for it in range(1, iterations + 1):
            if n_workers > 1:
                trajs, episode_log = collect_rollout_parallel(worker_procs, net)
            else:
                episode_log = []
                trajs, obs = collect_rollout(env, net, obs, rollout_steps, device, episode_log)

            stats = ppo_update(net, optimizer, trajs, device)

            for ep in episode_log:
                ep["iteration"] = it
                log_writer.writerow(ep)
                all_episode_scores.append(ep["score"])
            log_f.flush()

            if it % log_every == 0 or it == 1:
                recent = all_episode_scores[-20:] if all_episode_scores else [0.0]
                elapsed = wallclock.time() - t0
                print(
                    f"iter {it:4d}/{iterations}  episodes_so_far={len(all_episode_scores):4d}  "
                    f"mean_score(last20)={np.mean(recent):8.2f}  "
                    f"policy_loss={stats['policy_loss']:+.4f}  value_loss={stats['value_loss']:.4f}  "
                    f"entropy={stats['entropy']:.4f}  transitions={stats['n_transitions']:5d}  "
                    f"elapsed={elapsed:6.1f}s"
                )

            if it % checkpoint_every == 0 or it == iterations:
                ckpt_path = os.path.join(out_dir, f"{stage_name}_iter{it}.pt")
                torch.save(net.state_dict(), ckpt_path)
                print(f"  saved checkpoint: {ckpt_path}")
    finally:
        for w in worker_procs:
            w["in_q"].put(None)
        for w in worker_procs:
            w["process"].join(timeout=5)

    log_f.close()
    final_path = os.path.join(out_dir, f"{stage_name}_final.pt")
    torch.save(net.state_dict(), final_path)
    print(f"Training done. Final weights: {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Train one curriculum stage with IPPO.")
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="Checkpoint/log dir; default training/RL/checkpoints")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None,
                         help="Parallel rollout-collection processes. Default: min(4, cpu_count()-1). "
                              "1 disables multiprocessing entirely.")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(__file__), "checkpoints")
    train(
        stage_name=args.stage,
        iterations=args.iterations,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        out_dir=out_dir,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
