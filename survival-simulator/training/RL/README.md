# RL/

Curriculum-based RL training for the survival-simulator agent, built
entirely from `training/` -- `src/` is never modified. Same monkeypatch
pattern as `../rule_based/instrumentation.py`: rebind `Environment`
instance methods after construction rather than editing the class.

Full design rationale (why IPPO first, why the 5 curriculum stages are
bounded the way they are, why the reward function looks the way it does)
lives in Notion: **"NN based RL solution"**, with **"Curriculum"** as its
sub-page. This README documents what's actually implemented, in code, right
now.

## Files

- **`curriculum.py`** -- `StageConfig` dataclass + the 5-stage `STAGES`
  list (`stage1_foraging` -> `stage2_single_predator_evasion` ->
  `stage3_realistic_predator_pressure` -> `stage4_reproduction` ->
  `stage5_full_dynamics`). `apply_curriculum_patches(env, stage)` rebinds
  `env.spawn_predator` / `env.spawn_agent` on one `Environment` instance to
  turn predators off/cap them/leave them real, and to block
  reproduction/aging, per stage. Every stage shares the same observation
  and action schema (see `env_wrapper.py`), so a trained network's weights
  carry over stage-to-stage without re-initialization.

- **`reward.py`** -- `RewardState` + `attach_reward_tracking()` +
  `begin_tick()` / `compute_rewards()`. Dense per-agent reward:
  `w_energy * delta_energy/max_energy + w_survive * dt + w_repro * (had a
  child this tick) - w_death * (died this tick, any cause)`. Reward is
  privileged (can read anything off the env) and strictly separate from
  what the policy observes. Must be attached *after*
  `apply_curriculum_patches()` so it wraps whichever `spawn_agent`/
  `kill_agent` version the curriculum stage installed.

- **`env_wrapper.py`** -- `encode_observation()` turns one agent's
  `ObservationResponse`-shaped state dict into a fixed-length `OBS_DIM`
  (=72) vector: 7 own-scalar values (energy, age, speed, sprint_speed,
  max_energy, hearing_radius, vision_angle) + fixed K-slots per sensed
  object type (5 fruit, 5 agents, 3 predators, 3 trees, 3 edges),
  nearest-first, zero-padded/presence-flagged when fewer than K are
  sensed. `decode_action()` unscales a length-4 normalized `[-1, 1]` policy
  output into an `ActionRequest` (move_distance/move_direction/turn_angle/
  spawn_agent). `CurriculumEnv` wraps `SimulationCore` for one stage,
  gym-style `reset()` / `step(actions_norm) -> (obs, rewards, done, info)`.

  Only fields present in the real `ObservationResponse` contract
  (`src/utils/DTOs.py`) are ever read here -- no `sim_time`, no predator
  count, no privileged env state. See the Notion page's "What the agent
  actually has access to (hard constraint)" section.

  `CurriculumEnv._build()` (called by `reset()`) forces a `gc.collect()`
  before discarding the previous episode's environment -- see "Known issue
  found and fixed" below. Don't remove this; it's load-bearing, not
  defensive boilerplate.

- **`network.py`** -- `ActorCritic`: a small shared MLP trunk (2x128,
  tanh), a tanh-squashed Gaussian actor head over the 4-D action (mean
  bounded to [-1, 1], learnable state-independent log-std), and a linear
  critic head. One network, shared across every agent (IPPO). `.act()` for
  rollout collection (no-grad, returns both the clipped action to execute
  and the raw pre-clip action + log-prob to store for the PPO update);
  `.evaluate_actions()` for the update step itself.

- **`train.py`** -- the IPPO training loop: `collect_rollout()` tracks one
  trajectory per `agent_id` across ticks (IDs are never reused within an
  episode, so this is safe), finalizing each one either on death (true
  terminal, bootstrap value 0) or on truncation -- episode end while alive,
  or the rollout window closing mid-episode (bootstrap = the critic's own
  value estimate, so the value function isn't taught that hitting a time
  limit is as bad as dying). `ppo_update()` does standard clipped-surrogate
  PPO with GAE, run for a few epochs over shuffled minibatches. `train()` /
  `main()` wire it together with checkpointing (`<stage>_iter<N>.pt`,
  `<stage>_final.pt`) and a per-episode CSV log (`<stage>_episodes.csv`:
  iteration, score, sim_time, final_num_agents, extinct, wall_seconds).

  Usage:
  ```
  python3 training/RL/train.py --stage stage1_foraging --iterations 200
  ```
  Defaults to `training/RL/checkpoints/`; override with `--out`. Resume
  from a checkpoint (e.g. moving to the next curriculum stage) with
  `--resume-from <path>.pt --stage stage2_single_predator_evasion`.

## Known issue found and fixed: env-recreation memory leak

Every episode reset calls `SimulationCore.__init__` again, which rebuilds a
~1600x1200 `biome_map` and re-applies the curriculum/reward monkeypatches
to the new `Environment` instance. Those patches (`types.MethodType`-bound
hooks that close over the *previous* bound method) create a reference
cycle between the env and its own patched methods. Ordinary refcounting
can't break that cycle -- only Python's cyclic GC can -- and under a tight
training loop with short (stage-1, early-training) episodes, discarded
envs piled up faster than the default GC thresholds triggered a
collection. Measured effect: **real RSS growth of several hundred MB per
rollout iteration, OOM-killing the process (exit 137) within a couple of
minutes.** Confirmed by isolating the leak to the env layer alone (no
torch/PPO involved), then confirming an explicit `gc.collect()` per reset
holds memory flat. The fix is a single `gc.collect()` call at the top of
`CurriculumEnv._build()` -- already in `env_wrapper.py`. If you ever
rewrite this wrapper, keep it (or something equivalent); without it,
training WILL eventually get OOM-killed regardless of how much RAM is
available, it just takes longer on a bigger machine.

## Where training actually runs

`torch` couldn't be installed on the Windows machine's linked Linux VM
(the pip index PyTorch recommends, `download.pytorch.org`, isn't reachable
through that VM's network egress, and the plain PyPI `torch` package pulls
several GB of CUDA dependencies that don't fit in that VM's disk
allowance). Training was moved to Claude's own cloud workspace instead:
`src/` and `training/` are staged in from the connected folder, `torch`
(CPU build, via PyPI with `--no-deps` plus its real non-CUDA dependencies)
installs cleanly there, and checkpoints/logs get committed back to this
folder when a run is worth keeping. If you want to train directly on your
own machine instead, you'd need a CPU-only torch wheel obtained some other
way (e.g. `pip download` from a machine that CAN reach
`download.pytorch.org`, then transfer the wheel over).

## Validated: a real training run, end to end

A 60-iteration (512 steps/iteration, ~355 episodes) run of
`stage1_foraging` completed cleanly with the GC fix in place: memory
stayed flat (~0.9-1.2 GB, oscillating, not climbing) for the full ~340s
run, and mean score (last 20 episodes) moved from ~6.9 at iteration 1 to
~9.3-10.4 by iteration 50-60 -- a real, if modest, upward trend, which is
what "the plumbing works and *something* is learning" looks like this
early. Entropy stayed high and didn't visibly start dropping in just 60
iterations, so this is nowhere near converged -- it's a smoke test that
the full loop (env -> network -> PPO update -> improved policy) is
correct, not a trained agent yet.

## Not yet built

- Real training runs long enough to converge on each stage (this was a
  60-iteration correctness check, not a training budget).
- Stage-to-stage progression: training stage 2-5 in sequence via
  `--resume-from`, and deciding graduation criteria empirically rather
  than just by the Notion page's planned thresholds.
- Loading a trained checkpoint into `agent_server.py` at evaluation time.
- Hyperparameter tuning (reward weights, entropy coefficient, rollout
  length) if a stage's score plateaus somewhere it shouldn't.
- A resolution for the `CHUNK_SIZE`-adjacent normalization constants in
  `env_wrapper.py` (`hearing_radius` is currently scaled by a fixed
  constant rather than a value read off `chunk_size` -- fine for the
  default `chunk_size=400`, worth revisiting if that ever changes).
