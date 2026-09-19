"""
Reward computation for RL training: privileged, training-time-only, and
strictly separate from the observation the policy actually receives (see
env_wrapper.py's encode_observation, which only uses ObservationResponse
fields). Reward is free to look at anything on the Environment instance --
it never leaks into what the network sees.

Formula (decided, see the "Reward function (decided)" section of the
"NN based RL solution" Notion page):

    r_i = w_energy * delta_energy_i / max_energy_i
        + w_survive * dt
        + w_repro   * (1 if agent i had a child this tick else 0)
        - w_death   * (1 if agent i died this tick, any cause)

Death is cause-agnostic by design (see Notion page): the agent's job is to
not die, not to distinguish predator vs. starvation, and a single flat
penalty avoids the network having to learn two different "how bad was this"
scales for outcomes that are equally terminal.

Call order requirement: attach_reward_tracking() must run AFTER
apply_curriculum_patches() (curriculum.py), so this hook wraps whichever
spawn_agent/kill_agent version the curriculum stage installed -- otherwise a
curriculum-blocked spawn (e.g. reproduction disabled in stage 1-3) would
never reach the reward hook and a not-really-a-birth would get rewarded.
"""
import types
from dataclasses import dataclass, field
from typing import Dict, Optional


DEFAULT_WEIGHTS = dict(
    w_energy=1.0,
    w_survive=0.01,
    w_repro=0.4,
    w_death=7.0,
)


@dataclass
class RewardState:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    prev_energy: Dict[int, float] = field(default_factory=dict)
    died_this_tick: Dict[int, bool] = field(default_factory=dict)
    reproduced_this_tick: Dict[int, bool] = field(default_factory=dict)


def attach_reward_tracking(env, state: RewardState) -> None:
    """
    Hook kill_agent/spawn_agent on this ONE Environment instance so
    compute_rewards() can see, per tick, who died and who had a child.
    Cause-agnostic on purpose -- see module docstring.
    """
    original_kill_agent = env.kill_agent
    original_spawn_agent = env.spawn_agent

    def kill_agent_hook(self, agent):
        state.died_this_tick[agent.agent_id] = True
        return original_kill_agent(agent)

    def spawn_agent_hook(self, x=None, y=None, parent=None):
        child = original_spawn_agent(x=x, y=y, parent=parent)
        if child is not None and parent is not None:
            state.reproduced_this_tick[parent.agent_id] = True
        return child

    env.kill_agent = types.MethodType(kill_agent_hook, env)
    env.spawn_agent = types.MethodType(spawn_agent_hook, env)


def begin_tick(env, state: RewardState) -> None:
    """
    Call once, right before sim.step(actions), each tick:
      - snapshots each living agent's current energy (delta_energy is
        computed against this snapshot after the step)
      - clears the per-tick death/reproduction flags from the previous tick
    """
    state.prev_energy = {agent.agent_id: agent.energy for agent in env.agents}
    state.died_this_tick = {}
    state.reproduced_this_tick = {}


def compute_rewards(env, state: RewardState, dt: float) -> Dict[int, float]:
    """
    Call once, right after sim.step(actions), each tick. Returns a
    per-agent-id reward dict covering:
      - every agent still alive at the end of this tick, and
      - every agent that died during this tick (agent_id keys persist in
        state.died_this_tick / state.prev_energy even though the agent is
        no longer in env.agents by the time this runs).

    A dead agent's delta_energy term is skipped (there's no "current
    energy" to diff against once removed from the environment) -- the flat
    death penalty already covers that outcome.
    """
    w = state.weights
    rewards: Dict[int, float] = {}

    living_ids = set()
    for agent in env.agents:
        living_ids.add(agent.agent_id)
        prev = state.prev_energy.get(agent.agent_id, agent.energy)
        delta_energy = agent.energy - prev
        r = w["w_energy"] * (delta_energy / agent.max_energy)
        r += w["w_survive"] * dt
        if state.reproduced_this_tick.get(agent.agent_id):
            r += w["w_repro"]
        rewards[agent.agent_id] = r

    for agent_id, died in state.died_this_tick.items():
        if not died or agent_id in living_ids:
            continue
        r = rewards.get(agent_id, 0.0)
        r -= w["w_death"]
        rewards[agent_id] = r

    return rewards
