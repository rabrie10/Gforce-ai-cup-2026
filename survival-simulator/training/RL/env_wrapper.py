"""
Fixed-slot observation/action encoding + the CurriculumEnv wrapper that ties
together SimulationCore, curriculum.py's patches, and reward.py's tracking
-- all from training/, never touching src/.

Observation contract note (hard constraint, see Notion "What the agent
actually has access to" section): every field read here comes from
ObservationResponse (src/utils/DTOs.py) / env.get_agent_state()'s dict,
which is exactly what the real evaluator hands a deployed agent. No
sim_time, no predator count, no privileged env state -- if it's not in that
dict, the policy doesn't get to see it. Reward (reward.py) is the only
place privileged env state is allowed.

Each observed object in ObservationResponse.observations is a dict:
  {"type": "Fruit"|"Tree", "distance": float, "angle": float}
  {"type": "Agent"|"Predator", "distance": float, "angle": float, "rel_dir": float, ...}
  {"type": "Edge", "coords": ((sx, sy), (ex, ey))}   # agent-local frame, no distance/angle given directly
(src/elements/creature.py:Creature.observe -- the one function that builds
these for every observer, agent and predator alike.)
"""
import gc
import math
import random
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core import SimulationCore
from src.utils.DTOs import ActionRequest

from training.rule_based.instrumentation import patch_environment_for_headless
from training.RL.curriculum import StageConfig, apply_curriculum_patches
from training.RL.reward import RewardState, attach_reward_tracking, begin_tick, compute_rewards


# --- Fixed-slot sizes -------------------------------------------------------
K_FRUIT = 5
K_AGENT = 5
K_PREDATOR = 3
K_TREE = 3
K_EDGE = 3

# Own-scalar block: energy, age, speed, sprint_speed, max_energy, hearing_radius,
# vision_range, vision_angle -- all normalized to [0, 1]-ish. vision_range is
# needed because every object distance is divided by the agent's OWN
# vision_range (see encode_observation); without it absolute distances are
# unrecoverable once mutation makes vision_range differ between agents.
N_OWN = 8

# Per-slot field counts (this many normalized floats per slot, the LAST of
# which is always the presence flag -- 0.0 for a padded/empty slot).
FIELDS_FRUIT = 3     # distance, angle, presence
FIELDS_AGENT = 4      # distance, angle, rel_dir, presence
FIELDS_PREDATOR = 4   # distance, angle, rel_dir, presence
FIELDS_TREE = 3       # distance, angle, presence
FIELDS_EDGE = 3        # distance, angle, presence  (distance/angle derived from coords)

OBS_DIM = (
    N_OWN
    + K_FRUIT * FIELDS_FRUIT
    + K_AGENT * FIELDS_AGENT
    + K_PREDATOR * FIELDS_PREDATOR
    + K_TREE * FIELDS_TREE
    + K_EDGE * FIELDS_EDGE
)

# Normalization constants matching environment.py's own mutation clamps
# (src/elements/environment.py: spawn_agent's max_speed/max_sprint_speed/
# max_max_energy/max_hearing_radius/max_vision_radius/max_cone_angle). Speed
# and energy caps are fixed constants there, so they're safe to hardcode
# here too. hearing/vision caps are chunk_size-derived and MUST be read off
# the live env (self.sim.env.chunk_size) rather than hardcoded, since
# chunk_size is a SimulationCore constructor kwarg someone could change.
MAX_SPEED = 20.0
MAX_SPRINT_SPEED = 40.0
MAX_MAX_ENERGY = 1000.0
MAX_CONE_ANGLE = math.pi / 2
MAX_AGE_NORM = 180.0  # generous headroom over Agent's default 60-120 max_age range

# Distances/angles inside individual observations aren't bounded by a single
# constant (vision_range varies per-mutated-agent), so we normalize each
# object's distance by THIS agent's own vision_radius (its own maximum
# possible sensing distance) rather than a global constant.
MAX_ANGLE = math.pi  # angles from Creature.relative_distance_angle are wrapped to [-pi, pi]


def _closest_point_on_edge(coords: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
    """
    Same technique src/utils/controllers/heuristic_policy.py uses: given an
    edge's two endpoints in the agent-local frame (agent at origin, facing
    angle 0 -- exactly how environment.py hands them to us via the "Edge"
    observation), return (distance, angle) of the closest point on the
    segment to the agent.
    """
    (x1, y1), (x2, y2) = coords
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        px, py = x1, y1
    else:
        t = -(x1 * dx + y1 * dy) / denom
        t = max(0.0, min(1.0, t))
        px, py = x1 + t * dx, y1 + t * dy
    distance = math.hypot(px, py)
    angle = math.atan2(py, px)
    return distance, angle


def _fill_slots(items: List[Tuple[float, ...]], k: int, n_fields: int) -> List[float]:
    """
    items: list of tuples, each already normalized EXCEPT for the presence
    flag (n_fields - 1 values per item -- presence is appended here).
    Sorted nearest-first by the caller before this is called. Truncates to
    the k nearest, zero-pads (all fields 0.0, including presence) beyond
    that.
    """
    out: List[float] = []
    for i in range(k):
        if i < len(items):
            out.extend(items[i])
            out.append(1.0)  # presence
        else:
            out.extend([0.0] * (n_fields - 1))
            out.append(0.0)  # presence
    return out


def encode_observation(agent_state: dict) -> np.ndarray:
    """
    agent_state: the dict env.get_agent_state() returns / what
    ObservationResponse carries for one agent. Builds a fixed-length
    OBS_DIM vector: N_OWN own-scalar values, then K-slots per object type
    (Fruit, Agent, Predator, Tree, Edge), nearest-first, zero-padded.
    """
    own_vision_radius = max(agent_state["vision_range"], 1e-6)

    own = [
        agent_state["energy"] / max(agent_state["max_energy"], 1e-6),
        min(agent_state["age"] / MAX_AGE_NORM, 1.0),
        agent_state["speed"] / MAX_SPEED,
        agent_state["sprint_speed"] / MAX_SPRINT_SPEED,
        agent_state["max_energy"] / MAX_MAX_ENERGY,
        # hearing_radius/vision_range are per-agent mutated traits capped at
        # chunk_size-derived maxima (see src/elements/environment.py:
        # spawn_agent's max_hearing_radius = chunk_size/4). Normalizing
        # against the agent's OWN current vision_radius would always be a
        # no-op (self/self == 1), so instead scale by a generous fixed
        # constant (200, comfortably above the default hearing_radius=50
        # and the observed chunk_size/4=100 cap) and clip to 1.0.
        min(agent_state["hearing_radius"] / 200.0, 1.0),
        min(agent_state["vision_range"] / 200.0, 1.0),
        agent_state["vision_angle"] / MAX_CONE_ANGLE,
    ]

    observations = agent_state.get("observations", [])

    fruits, agents, predators, trees, edges = [], [], [], [], []
    for obs in observations:
        t = obs.get("type")
        if t == "Fruit":
            fruits.append(obs)
        elif t == "Agent":
            agents.append(obs)
        elif t == "Predator":
            predators.append(obs)
        elif t == "Tree":
            trees.append(obs)
        elif t == "Edge":
            edges.append(obs)

    def norm_da(o):
        return (
            min(o["distance"] / own_vision_radius, 1.0),
            o["angle"] / MAX_ANGLE,
        )

    fruit_items = sorted((norm_da(o) for o in fruits), key=lambda t: t[0])
    tree_items = sorted((norm_da(o) for o in trees), key=lambda t: t[0])

    def norm_da_dir(o):
        d, a = norm_da(o)
        return (d, a, o.get("rel_dir", 0.0) / MAX_ANGLE)

    agent_items = sorted((norm_da_dir(o) for o in agents), key=lambda t: t[0])
    predator_items = sorted((norm_da_dir(o) for o in predators), key=lambda t: t[0])

    edge_da = []
    for o in edges:
        d, a = _closest_point_on_edge(o["coords"])
        edge_da.append((min(d / own_vision_radius, 1.0), a / MAX_ANGLE))
    edge_items = sorted(edge_da, key=lambda t: t[0])

    vec = (
        own
        + _fill_slots(fruit_items, K_FRUIT, FIELDS_FRUIT)
        + _fill_slots(agent_items, K_AGENT, FIELDS_AGENT)
        + _fill_slots(predator_items, K_PREDATOR, FIELDS_PREDATOR)
        + _fill_slots(tree_items, K_TREE, FIELDS_TREE)
        + _fill_slots(edge_items, K_EDGE, FIELDS_EDGE)
    )
    arr = np.asarray(vec, dtype=np.float32)
    assert arr.shape[0] == OBS_DIM, f"encode_observation produced {arr.shape[0]}, expected {OBS_DIM}"
    return arr


def decode_action(agent_id: int, action_norm: np.ndarray, sprint_speed_own: float) -> ActionRequest:
    """
    action_norm: length-4 array in [-1, 1] (policy network output, e.g. via
    tanh on the first 3 and a raw logit on the 4th):
      [0] move_distance_norm  -> scaled to [0, sprint_speed_own]  (per-tick distance cap)
      [1] move_direction_norm -> scaled to [-pi, pi]
      [2] turn_angle_norm     -> scaled to [-pi, pi]
      [3] spawn_logit         -> spawn_agent = spawn_logit > 0.5
    """
    move_distance = max(0.0, (float(action_norm[0]) + 1.0) / 2.0) * sprint_speed_own
    move_direction = float(action_norm[1]) * math.pi
    turn_angle = float(action_norm[2]) * math.pi
    spawn_agent = bool(action_norm[3] > 0.5)
    return ActionRequest(
        agent_id=agent_id,
        move_distance=move_distance,
        move_direction=move_direction,
        turn_angle=turn_angle,
        spawn_agent=spawn_agent,
    )


class CurriculumEnv:
    """
    Wraps SimulationCore for one curriculum stage: applies
    apply_curriculum_patches() + initial-population aging override +
    attach_reward_tracking(), and exposes a gym-ish reset()/step() that
    speaks fixed-length observation vectors and normalized actions.
    """

    def __init__(self, stage: StageConfig, seed: Optional[int] = None, reward_weights: Optional[dict] = None):
        self.stage = stage
        self.seed = seed
        # Persistent, seeded-once RNG that DRAWS a fresh seed for
        # SimulationCore on every _build() call (see _build()'s comment --
        # reusing self.seed directly on every reset() meant every episode
        # replayed the exact same biome map / fruit / obstacle / predator
        # RNG stream, which is an overfitting risk, not a feature). Keeping
        # this RNG seeded from the top-level `seed` still makes a whole
        # training run reproducible end to end when you pass the same seed.
        self._episode_rng = random.Random(seed)
        self.reward_state = RewardState()
        # Stage-level weight overrides (curriculum.py's StageConfig.reward_weights
        # -- e.g. stage 2/3/5 turning on w_danger evasion shaping) apply first,
        # so an explicit reward_weights argument here can still override them
        # per-call if ever needed (train.py doesn't currently pass one; it
        # relies entirely on the stage config).
        if stage.reward_weights:
            self.reward_state.weights.update(stage.reward_weights)
        if reward_weights:
            self.reward_state.weights.update(reward_weights)
        self.sim: Optional[SimulationCore] = None
        self._last_state: Optional[dict] = None
        self._build()
        # __init__ already built episode #1; the first reset() must reuse it
        # instead of building (and drawing seeds for) a second env.
        self._fresh = True

    def _build(self) -> None:
        # Every monkeypatch below (spawn_predator/spawn_agent/kill_agent
        # rebound via types.MethodType) creates a reference cycle: the env
        # holds the bound patched method, whose closure captures the
        # "original" bound method, which itself references the env. Regular
        # refcounting can't break that cycle -- only the cyclic GC can, and
        # under a tight training loop with short (~seconds-long) episodes,
        # a full env (including its ~1600x1200 biome_map) can pile up
        # faster than Python's default gc thresholds trigger a collection,
        # causing real, measured multi-GB/minute RSS growth. Forcing a
        # collection right before discarding the previous env keeps this
        # bounded. (Confirmed by direct measurement: RSS grew unboundedly
        # over a few thousand ticks without this, and stayed flat with it.)
        gc.collect()
        patch_environment_for_headless()

        # Fresh seed EVERY episode (drawn from the persistent, top-level-
        # seeded _episode_rng) -- see __init__'s comment. Without this,
        # every episode in this CurriculumEnv replayed an identical map.
        episode_seed = self._episode_rng.randint(0, 2**32 - 1)

        env_kwargs = dict(self.stage.env_kwargs)
        if self.stage.vary_starting_fruits:
            # SimulationCore's own default is env_width // 50 (see
            # src/core.py) -- vary around that instead of a magic number,
            # so this still makes sense if env_width is ever changed.
            default_fruits = max(4, 1600 // 50)
            env_kwargs["starting_fruits"] = self._episode_rng.randint(
                max(4, default_fruits // 2), default_fruits * 2
            )

        self.sim = SimulationCore(seed=episode_seed, dt=1 / 10, **env_kwargs)
        env = self.sim.env

        # Curriculum patches (spawn_predator/spawn_agent) apply from this
        # point forward -- affects predators/agents created after this call,
        # not the initial population create_environment() already spawned.
        apply_curriculum_patches(env, self.stage)

        # Aging-off for the INITIAL population: those agents were created
        # inside create_environment() before we had the instance back, so
        # disabling aging for them means setting max_age directly here
        # rather than patching a method (curriculum.py's docstring points
        # here for exactly this reason).
        if not self.stage.aging:
            for agent in env.agents:
                agent.max_age = float("inf")

        # attach_reward_tracking() must run AFTER apply_curriculum_patches()
        # so it wraps whichever spawn_agent/kill_agent version the
        # curriculum stage installed (reward.py's module docstring).
        attach_reward_tracking(env, self.reward_state)

        # Per-episode accumulators (fresh every _build(), i.e. every episode).
        # Kept HERE, on the env, instead of as locals inside train.py's
        # collect_rollout: an episode (up to 3000 ticks) spans many 512-tick
        # rollout calls, and a per-call local silently undercounted it.
        self.episode_stats = {"fruit_energy": 0.0, "predator_deaths": 0, "starvation_deaths": 0}

    def global_features(self) -> np.ndarray:
        """Colony-level state for the centralized critic (privileged; never given to the actor)."""
        env = self.sim.env
        ags = env.agents
        n = len(ags)
        if n:
            ef = [a.energy / a.max_energy if a.max_energy > 0 else 0.0 for a in ags]
            mean_ef, min_ef = float(np.mean(ef)), float(np.min(ef))
            mean_age = float(np.mean([a.age for a in ags])) / 100.0
        else:
            mean_ef = min_ef = mean_age = 0.0
        return np.array([
            n / 40.0, mean_ef, min_ef,
            len(env.fruits) / 60.0, len(env.trees) / 20.0, len(env.predators) / 6.0,
            env.time / max(self.stage.max_sim_time, 1e-6), mean_age,
        ], dtype=np.float32)

    def reset(self) -> Dict[int, np.ndarray]:
        if self._fresh:
            self._fresh = False
        else:
            self._build()
        return self._current_observations()

    def _current_observations(self) -> Dict[int, np.ndarray]:
        env = self.sim.env
        obs = {}
        for agent in env.agents:
            state = env.get_agent_state(agent.agent_id)
            obs[agent.agent_id] = encode_observation(state)
        return obs

    def step(self, actions_norm: Dict[int, np.ndarray]):
        """
        actions_norm: {agent_id: length-4 normalized action array}, one
        entry per currently-living agent (missing agent_ids are simply
        skipped this tick, same as run_episode.py's pattern).

        Returns (observations, rewards, done, info) where:
          observations: {agent_id: OBS_DIM vector} for agents alive AFTER the step
          rewards:      {agent_id: float} for agents alive before the step
                        (includes agents that died this tick, one final reward)
          done:         True if the colony is extinct or max_sim_time reached
          info:         {"sim_time", "score", "num_agents"}
        """
        env = self.sim.env
        begin_tick(env, self.reward_state)

        actions = []
        for agent in env.agents:
            a_norm = actions_norm.get(agent.agent_id)
            if a_norm is None:
                continue
            actions.append((agent.agent_id, decode_action(agent.agent_id, a_norm, agent.sprint_speed)))

        state = self.sim.step(actions)
        rewards = compute_rewards(env, self.reward_state, self.sim.dt)
        observations = self._current_observations()

        done = (len(env.agents) == 0) or (env.time >= self.stage.max_sim_time)

        st = self.episode_stats
        st["fruit_energy"] += self.reward_state.last_fruit_energy_tick
        st["predator_deaths"] += self.reward_state.predator_deaths_this_tick
        st["starvation_deaths"] += self.reward_state.starvation_deaths_this_tick

        info = {
            "sim_time": env.time,
            "score": env.score,
            "num_agents": len(env.agents),
            # Food-consumption diagnostic (reward.py's algebraic
            # decomposition). Not a reward term.
            "fruit_energy_tick": self.reward_state.last_fruit_energy_tick,
            # Whole-episode running totals (correct across rollout-window
            # boundaries) -- train.py logs these when done=True.
            "episode_stats": dict(st),
        }
        self._last_state = state
        return observations, rewards, done, info
