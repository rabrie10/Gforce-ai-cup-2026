"""
Curriculum stage definitions for RL training: env kwargs + which monkeypatches
apply, so the same fixed observation/action schema trains through all 5 stages
without ever touching src/. See the "Curriculum" Notion sub-page (under
"NN based RL solution") for the full rationale behind each stage's boundaries
and graduation criteria; this file is the executable form of that plan.

Every patch here rebinds a method on ONE Environment instance after it's
constructed -- same pattern as training/rule_based/instrumentation.py. The
one exception is "aging off" for the INITIAL population: those agents are
created inside SimulationCore's own constructor, before we get the instance
back, so disabling aging for them means setting agent.max_age directly after
construction, not patching a method. That construction step lives in
CurriculumEnv.__init__ in env_wrapper.py, not in this file -- this module
only defines the stage table and the post-construction patches.
"""
import types
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StageConfig:
    name: str
    description: str
    env_kwargs: dict
    predators: str  # "off" | "capped" | "real"
    predator_cap: Optional[int] = None
    reproduction: bool = True
    aging: bool = True
    max_sim_time: float = 3000.0
    vary_starting_fruits: bool = False


STAGES = [
    StageConfig(
        name="stage1_foraging",
        description=(
            "Efficient navigation to sensed fruit, walk/sprint energy tradeoff, "
            "obstacle avoidance. No threats, no population dynamics -- isolates "
            "the one skill that dominates the score decomposition (survival "
            "time) before anything else is introduced."
        ),
        env_kwargs=dict(starting_agents=3, starting_predators=0),
        predators="off",
        reproduction=False,
        aging=False,
        max_sim_time=300.0,
    ),
    StageConfig(
        name="stage2_single_predator_evasion",
        description=(
            "Evade exactly one predator while continuing to forage. Capped at "
            "one so any death during this stage has an unambiguous cause, same "
            "clean-attribution property the rule_based death-cause logging relies on."
        ),
        env_kwargs=dict(starting_agents=3, starting_predators=1),
        predators="capped",
        predator_cap=1,
        reproduction=False,
        aging=False,
        max_sim_time=400.0,
    ),
    StageConfig(
        name="stage3_realistic_predator_pressure",
        description=(
            "Real time-scaling predator spawn formula, no synthetic cap -- the "
            "actual in-game dynamic, not a stand-in. Reproduction still off: "
            "this stage is 'cope with real threat density using the moves "
            "already learned', not a new decision type yet."
        ),
        env_kwargs=dict(starting_agents=3, starting_predators=0),
        predators="real",
        reproduction=False,
        aging=False,
        max_sim_time=600.0,
    ),
    StageConfig(
        name="stage4_reproduction",
        description=(
            "Real aging and reproduction enabled on top of an already-competent "
            "forage+evade policy -- a new action type layered on late, not "
            "learned simultaneously with everything else."
        ),
        env_kwargs=dict(starting_agents=5, starting_predators=0),
        predators="real",
        reproduction=True,
        aging=True,
        max_sim_time=1200.0,
    ),
    StageConfig(
        name="stage5_full_dynamics",
        description=(
            "Matches the real evaluator exactly -- no patches at all. Also "
            "varies starting_fruits episode-to-episode (folded in from the "
            "original 'robustness' stage idea) to avoid overfitting to one "
            "food-density regime. Doubles as the ongoing benchmark."
        ),
        env_kwargs=dict(starting_agents=5, starting_predators=0),
        predators="real",
        reproduction=True,
        aging=True,
        max_sim_time=3000.0,
        vary_starting_fruits=True,
    ),
]

_STAGES_BY_NAME = {s.name: s for s in STAGES}


def get_stage(name: str) -> StageConfig:
    if name not in _STAGES_BY_NAME:
        raise KeyError(f"Unknown stage {name!r}. Known stages: {list(_STAGES_BY_NAME)}")
    return _STAGES_BY_NAME[name]


def apply_curriculum_patches(env, stage: StageConfig) -> None:
    """
    Rebind env.spawn_predator and/or env.spawn_agent on this ONE Environment
    instance per the stage config. Call this AFTER the environment (and its
    initial population) already exists -- it only affects predators/agents
    created from this point forward. For "aging off" on the INITIAL
    population, see CurriculumEnv.__init__ in env_wrapper.py instead --
    that's where the initial agents' max_age gets overridden directly.
    """
    original_spawn_predator = env.spawn_predator
    original_spawn_agent = env.spawn_agent

    if stage.predators == "off":
        def spawn_predator_hook(self, *a, **kw):
            return None
        env.spawn_predator = types.MethodType(spawn_predator_hook, env)

    elif stage.predators == "capped":
        cap = stage.predator_cap or 1

        def spawn_predator_hook(self, *a, **kw):
            if len(self.predators) >= cap:
                return None
            return original_spawn_predator(*a, **kw)
        env.spawn_predator = types.MethodType(spawn_predator_hook, env)

    # stage.predators == "real": leave spawn_predator untouched.

    if not stage.reproduction:
        def spawn_agent_hook(self, x=None, y=None, parent=None):
            if parent is not None:
                return None  # block births; initial seeding (parent=None) unaffected
            return original_spawn_agent(x=x, y=y, parent=parent)
        env.spawn_agent = types.MethodType(spawn_agent_hook, env)

    if not stage.aging:
        # Wraps whichever spawn_agent is currently bound (possibly the
        # reproduction-blocking hook just above), so both patches compose.
        # In today's 5 stages, aging=False always pairs with reproduction=False,
        # so this only matters if a future stage mixes them differently --
        # kept general rather than relying on that coupling.
        current_spawn_agent = env.spawn_agent

        def spawn_agent_infinite_age_hook(self, x=None, y=None, parent=None):
            agent = current_spawn_agent(x=x, y=y, parent=parent)
            if agent is not None:
                agent.max_age = float("inf")
            return agent
        env.spawn_agent = types.MethodType(spawn_agent_infinite_age_hook, env)
