"""
Reward computation for RL training: privileged, training-time-only, and
strictly separate from the observation the policy actually receives (see
env_wrapper.py's encode_observation, which only uses ObservationResponse
fields). Reward is free to look at anything on the Environment instance --
it never leaks into what the network sees, with ONE deliberate exception:
the evasion-shaping term below reads env.agent_observations, which is
exactly the same perception the policy itself gets (nothing privileged),
by design -- see "Evasion shaping" below for why that matters.

Base formula (per agent, per tick):

    r_i = w_energy * delta_energy_i / max_energy_i
        + w_survive * dt
        + w_repro   * (1 if agent i had a child this tick else 0)
        - w_death   * (1 if agent i died this tick, any cause)
        + w_danger  * (gamma * Phi(s') - Phi(s))      [only if w_danger != 0]

Death is cause-agnostic by design: the agent's job is to not die, not to
distinguish predator vs. starvation, and a single flat penalty avoids the
network having to learn two different "how bad was this" scales for
outcomes that are equally terminal.

Evasion shaping (the w_danger term)
------------------------------------
Stages 2, 3, and 5's training plateaued at ~90-100% extinction with no real
evasion learning. Root cause: w_death is a flat penalty paid ONCE, at the
moment of death -- there's no signal during the 50+ ticks leading up to
that death saying "the predator getting closer right now is bad". Energy
delta gives foraging a dense per-tick signal; evasion had nothing
equivalent.

Phi(s) is a "danger potential": 0 when no predator is perceived (or the
nearest one is beyond DANGER_RADIUS), ramping linearly down to -1 as the
nearest PERCEIVED predator's distance goes to 0. "Perceived" is load-
bearing: Phi only ever looks at env.agent_observations, the same dict
encode_observation() reads to build the policy's own input -- never the
full privileged predator list -- so the shaping signal stays something the
agent could plausibly have learned to react to, not free information.

Adding gamma*Phi(s') - Phi(s) to the reward each tick is potential-based
reward shaping (Ng, Harada, Russell 1999): for ANY potential function Phi,
this transformation provably does not change which policy is optimal, it
only changes how fast an RL algorithm can find it by turning a sparse
terminal signal into a dense per-tick one. w_danger=0.0 (the default)
disables this term entirely and reproduces the original stage-1 formula
exactly -- see curriculum.py's per-stage `reward_weights` for which stages
turn it on.

Food-consumption tracking
--------------------------
Environment.score has exactly three additive terms per tick (see
../rule_based/instrumentation.py's docstring, which established this same
decomposition for the heuristic sweep):

    score += dt                          (once per tick, unconditionally)
    score += fruit.energy / 1000          (per fruit eaten, inline in the agent loop)
    score -= agent.energy / 100           (per agent eaten by a predator)

The predation term is captured exactly (not estimated) by the kill_agent
hook below, using the same energy>0-at-kill-time trick instrumentation.py
uses to classify predator vs. starvation deaths deterministically. That
makes the fruit term the only unknown in the equation, so it's solved for
algebraically every tick:

    fruit_score_this_tick   = score_delta - dt + predation_penalty_this_tick
    fruit_energy_this_tick  = fruit_score_this_tick * 1000   (undo the /1000)

Exposed via state.last_fruit_energy_tick each tick (env_wrapper.py folds it
into step()'s info dict; train.py accumulates it per episode into the CSV
log as fruit_energy_consumed). This is logged as a DIAGNOSTIC metric only --
it is deliberately NOT added as its own reward term. w_energy already
rewards eating fruit indirectly (energy goes up when you eat), so a second
explicit "fruit eaten" bonus on top would double-count the same behavior
and risk skewing the forage/evade tradeoff. Revisit this if the logged
numbers suggest foraging is being neglected once evasion shaping is on.

Call order requirement: attach_reward_tracking() must run AFTER
apply_curriculum_patches() (curriculum.py), so this hook wraps whichever
spawn_agent/kill_agent version the curriculum stage installed -- otherwise a
curriculum-blocked spawn (e.g. reproduction disabled in stage 1-3) would
never reach the reward hook and a not-really-a-birth would get rewarded.
"""
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional


DEFAULT_WEIGHTS = dict(
    w_energy=1.0,
    w_survive=0.01,
    w_repro=0.4,
    w_death=7.0,
    w_danger=0.0,  # off by default; curriculum.py turns this on per-stage
)

# World-unit radius within which the evasion-shaping potential ramps from 0
# (no danger) to -1 (predator right on top of the agent). Predators become
# perceivable (hearing) around distance ~50 and (vision) out to roughly
# vision_radius (~100-200 depending on mutation), so 150 puts the ramp
# comfortably inside the range agents can actually react to. Tune this if
# the shaped stages still don't show dense evasion behavior emerging.
DANGER_RADIUS = 150.0

# Discount used inside the shaping term itself (Ng et al.'s gamma*Phi(s') -
# Phi(s)). Should match train.py's GAMMA (0.99) -- kept as a separate
# constant here rather than imported, since reward.py has no reason to
# depend on train.py and the two are unlikely to drift given how rarely
# either changes.
SHAPING_GAMMA = 0.99


def _danger_potential(observation: Optional[List[dict]]) -> float:
    """
    Phi(s): 0 if no Predator entry is in this agent's own last observation
    (or the nearest one is beyond DANGER_RADIUS), otherwise
    -clip((DANGER_RADIUS - nearest_distance) / DANGER_RADIUS, 0, 1).
    `observation` is the flat list env.agent_observations[agent_id] holds
    (see src/elements/environment.py get_agent_state) -- exactly what the
    policy itself perceives, nothing more.
    """
    if not observation:
        return 0.0
    nearest = None
    for o in observation:
        if o.get("type") != "Predator":
            continue
        d = o.get("distance")
        if d is not None and (nearest is None or d < nearest):
            nearest = d
    if nearest is None:
        return 0.0
    return -max(0.0, min(1.0, (DANGER_RADIUS - nearest) / DANGER_RADIUS))


# Which predator information Phi uses:
#   "perceived" (default, original behaviour): only predators in the agent's
#       OWN last observation (vision cone / hearing radius).
#   "true": the nearest AWAKE predator by true distance (privileged, reward-
#       only -- it never enters the observation). Still a function of the full
#       simulator state, so the Ng et al. invariance argument still holds. Use
#       it if predators flanking out of the agent's vision cone make the
#       perceived Phi jump to 0 while the threat is still closing in (see
#       reward_sanity.py's popIn/popOut columns and encounter test).
PHI_MODE = "perceived"


def _danger_potential_true(env, agent) -> float:
    nearest = None
    for p in env.predators:
        if getattr(p, "resting", False):
            continue  # a sleeping predator is not a threat
        d = ((p.x - agent.x) ** 2 + (p.y - agent.y) ** 2) ** 0.5
        if nearest is None or d < nearest:
            nearest = d
    if nearest is None:
        return 0.0
    return -max(0.0, min(1.0, (DANGER_RADIUS - nearest) / DANGER_RADIUS))


def _phi(env, agent, observation) -> float:
    if PHI_MODE == "true":
        return _danger_potential_true(env, agent)
    return _danger_potential(observation)


@dataclass
class RewardState:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    prev_energy: Dict[int, float] = field(default_factory=dict)
    prev_observations: Dict[int, List[dict]] = field(default_factory=dict)
    prev_phi: Dict[int, float] = field(default_factory=dict)  # Phi(s) snapshot from begin_tick()
    died_this_tick: Dict[int, bool] = field(default_factory=dict)
    reproduced_this_tick: Dict[int, bool] = field(default_factory=dict)
    prev_score: float = 0.0
    predation_energy_this_tick: float = 0.0
    last_fruit_energy_tick: float = 0.0
    # Per-tick death counts by cause (exact -- same energy>0-at-kill-time
    # classification the predation-energy tracking uses). Reset every
    # begin_tick(); env_wrapper.py folds them into per-episode totals.
    predator_deaths_this_tick: int = 0
    starvation_deaths_this_tick: int = 0


def attach_reward_tracking(env, state: RewardState) -> None:
    """
    Hook kill_agent/spawn_agent on this ONE Environment instance so
    compute_rewards() can see, per tick, who died, who had a child, and how
    much energy was lost to predation (for the fruit-tracking algebra
    above). Death itself stays cause-agnostic for the reward's own
    -w_death term -- see module docstring -- but the predation ENERGY
    total is still needed to solve for fruit consumption, so it's captured
    here regardless.

    Cause classification for that energy total reuses
    ../rule_based/instrumentation.py's exact trick: kill_agent is only ever
    called with agent.energy <= 0 for a starvation death, or > 0 for a
    predator-caused one (the predator loop can only reach agents that
    haven't already starved out this tick) -- deterministic, not a guess.
    """
    original_kill_agent = env.kill_agent
    original_spawn_agent = env.spawn_agent

    def kill_agent_hook(self, agent):
        state.died_this_tick[agent.agent_id] = True
        if agent.energy > 0:
            state.predation_energy_this_tick += agent.energy
            state.predator_deaths_this_tick += 1
        else:
            state.starvation_deaths_this_tick += 1
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
      - snapshots each living agent's current observation list (Phi(s) for
        the evasion-shaping term is computed against this "before" snapshot)
      - snapshots env.score (for the fruit-consumption algebra)
      - clears the per-tick death/reproduction/predation-energy trackers
        from the previous tick
    """
    state.prev_energy = {agent.agent_id: agent.energy for agent in env.agents}
    state.prev_observations = {
        agent.agent_id: list(env.agent_observations.get(agent.agent_id, []))
        for agent in env.agents
    }
    state.prev_phi = {
        agent.agent_id: _phi(env, agent, state.prev_observations.get(agent.agent_id))
        for agent in env.agents
    } if state.weights.get("w_danger", 0.0) else {}
    state.prev_score = env.score
    state.died_this_tick = {}
    state.reproduced_this_tick = {}
    state.predation_energy_this_tick = 0.0
    state.predator_deaths_this_tick = 0
    state.starvation_deaths_this_tick = 0


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
    death penalty already covers that outcome. For a dead agent, the
    evasion-shaping term treats Phi(s') as 0 (the standard convention for
    an absorbing terminal state in potential-based shaping), so the shaped
    term there is just -w_danger * Phi(s_prev).

    Also updates state.last_fruit_energy_tick as a side effect (the fruit-
    consumption diagnostic -- see module docstring); env_wrapper.py's
    step() reads it into info["fruit_energy_tick"] right after this call.
    """
    w = state.weights
    rewards: Dict[int, float] = {}

    # --- fruit-consumption algebra (diagnostic only, not a reward term) ---
    score_delta = env.score - state.prev_score
    fruit_score = score_delta - dt + (state.predation_energy_this_tick / 100.0)
    state.last_fruit_energy_tick = max(0.0, fruit_score) * 1000.0

    w_danger = w.get("w_danger", 0.0)

    living_ids = set()
    for agent in env.agents:
        living_ids.add(agent.agent_id)
        prev = state.prev_energy.get(agent.agent_id, agent.energy)
        delta_energy = agent.energy - prev
        r = w["w_energy"] * (delta_energy / agent.max_energy)
        r += w["w_survive"] * dt
        if state.reproduced_this_tick.get(agent.agent_id):
            r += w["w_repro"]

        if w_danger:
            phi_prev = state.prev_phi.get(agent.agent_id, 0.0)
            phi_curr = _phi(env, agent, env.agent_observations.get(agent.agent_id, []))
            r += w_danger * (SHAPING_GAMMA * phi_curr - phi_prev)

        rewards[agent.agent_id] = r

    for agent_id, died in state.died_this_tick.items():
        if not died or agent_id in living_ids:
            continue
        r = rewards.get(agent_id, 0.0)
        r -= w["w_death"]
        if w_danger:
            phi_prev = state.prev_phi.get(agent_id, 0.0)
            r += w_danger * (SHAPING_GAMMA * 0.0 - phi_prev)
        rewards[agent_id] = r

    return rewards
