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
from typing import List, Optional, Tuple

# Heuristic baseline this whole curriculum is measured against (see
# training/rule_based/results/sweep_summary.csv): mean 511.8s / median
# 507.1s to extinction over a 30-seed sweep. Stage 3+'s goal-based stopping
# criteria are pinned relative to this number rather than an absolute
# survive-to-cap bar -- see StageConfig.success_metric's docstring for why.
HEURISTIC_MEAN_EXTINCTION_S = 511.8


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

    # Per-stage reward-weight overrides, merged onto reward.DEFAULT_WEIGHTS
    # (only keys present here are changed -- see env_wrapper.py's
    # CurriculumEnv.__init__, which does `reward_state.weights.update(...)`).
    # None means "use the defaults unchanged".
    reward_weights: Optional[dict] = None

    # Goal-based training stop, replacing a fixed --iterations count as the
    # PRIMARY stop condition (--iterations remains as a hard safety cap so a
    # run can't loop forever if the goal turns out unreachable). Checked
    # against a rolling window of the last `success_window` completed
    # episodes, once that many have accumulated.
    #
    # success_metric one of:
    #   "survive_rate"           -- fraction of the window's episodes that
    #                               end with >=1 agent still alive (not
    #                               extinct). Stage 1's own criterion
    #                               ("90% of episodes have at least 1 of
    #                               the 3 agents outlive the simulation")
    #                               is exactly this metric at threshold=0.90.
    #   "mean_sim_time_ratio"    -- (mean sim_time over the window) /
    #                               success_baseline, checked against
    #                               threshold. Used for stage 3+ instead of
    #                               a survive-to-cap percentage, because
    #                               stage 3's predator spawn rate is
    #                               uncapped and grows with sim_time --
    #                               "100% survive to the 600s/3000s cap"
    #                               may not be an achievable bar at ANY
    #                               policy quality. Benchmarking against
    #                               the heuristic's own recorded mean
    #                               instead gives a bar known to be
    #                               reachable (the heuristic reaches it)
    #                               and directly answers "did the RL
    #                               policy catch up".
    #   "population_growth_rate" -- fraction of the window's episodes that
    #                               end with MORE agents alive than the
    #                               stage started with (reproduction net
    #                               positive), not just "didn't go extinct".
    success_metric: Optional[str] = None
    success_threshold: float = 0.9
    success_window: int = 50
    success_baseline: Optional[float] = None  # required for "mean_sim_time_ratio"


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
        # Your own criterion, verbatim: train until >=90% of a 50-episode
        # window has at least 1 of the 3 starting agents alive at the 300s
        # cap. No predators in this stage, so no danger-shaping weight.
        success_metric="survive_rate",
        success_threshold=0.90,
        success_window=50,
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
        # Evasion-shaping turned on (see reward.py's module docstring for
        # why: this is the stage that plateaued hardest without it). 75%
        # survive-to-cap, not 90% -- evading a real predator is a harder
        # bar than pure foraging; flag if you want this tighter or looser.
        reward_weights=dict(w_danger=1.0),
        success_metric="survive_rate",
        success_threshold=0.75,
        success_window=50,
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
        # Uncapped predator growth -- see success_metric's docstring for
        # why this isn't a survive-to-cap percentage. Goal: mean sim_time
        # over the window reaches 85% of the heuristic's own 511.8s mean.
        # Threshold is your call; 85% (not 100%) leaves room for this
        # stage's bar to be "meaningfully competent", with stage 5 as the
        # one that actually has to match or beat the heuristic outright.
        reward_weights=dict(w_danger=1.0),
        success_metric="mean_sim_time_ratio",
        success_baseline=HEURISTIC_MEAN_EXTINCTION_S,
        success_threshold=0.85,
        success_window=50,
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
        # w_repro trimmed from the 0.4 default: reproduction already costs
        # the agent 100 energy directly (env_wrapper.py's agent_step), so
        # the extra reward bonus on top risks over-incentivizing rapid
        # breeding over staying alive. Goal: net population growth (more
        # agents at the end than the 5 it started with) in >=60% of the
        # window's episodes -- open to a different bar.
        reward_weights=dict(w_danger=1.0, w_repro=0.2),
        success_metric="population_growth_rate",
        success_threshold=0.60,
        success_window=50,
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
        # The real benchmark: matches the evaluator exactly, so the goal
        # here is the actual submission bar -- mean sim_time over the
        # window has to reach (not just approach) the heuristic's 511.8s
        # mean. threshold=1.0 means ratio >= 1.0, i.e. beat or match it.
        reward_weights=dict(w_danger=1.0, w_repro=0.2),
        success_metric="mean_sim_time_ratio",
        success_baseline=HEURISTIC_MEAN_EXTINCTION_S,
        success_threshold=1.0,
        success_window=50,
    ),
]

_STAGES_BY_NAME = {s.name: s for s in STAGES}


def get_stage(name: str) -> StageConfig:
    if name not in _STAGES_BY_NAME:
        raise KeyError(f"Unknown stage {name!r}. Known stages: {list(_STAGES_BY_NAME)}")
    return _STAGES_BY_NAME[name]


def check_goal(stage: StageConfig, episodes: List[dict]) -> Tuple[bool, Optional[float]]:
    """
    Rolling-window goal check. `episodes` is the FULL chronological list of
    completed-episode dicts this run has logged so far (train.py's
    all_episodes list -- same shape as the *_episodes.csv rows, plus
    starting_agents/fruit_energy_consumed). Returns (goal_met, current_rate):

      - (False, None) if the stage has no success_metric configured, or
        fewer than success_window episodes have completed yet (not enough
        data for a stable rate -- avoids stopping on a lucky first 3
        episodes).
      - Otherwise (rate >= success_threshold, rate), computed over exactly
        the last success_window episodes.
    """
    if not stage.success_metric or len(episodes) < stage.success_window:
        return False, None

    window = episodes[-stage.success_window:]

    if stage.success_metric == "survive_rate":
        rate = sum(1 for e in window if not e["extinct"]) / len(window)

    elif stage.success_metric == "mean_sim_time_ratio":
        if not stage.success_baseline:
            raise ValueError(f"{stage.name}: success_metric='mean_sim_time_ratio' needs success_baseline set")
        mean_sim_time = sum(e["sim_time"] for e in window) / len(window)
        rate = mean_sim_time / stage.success_baseline

    elif stage.success_metric == "population_growth_rate":
        rate = sum(
            1 for e in window
            if e["final_num_agents"] > e.get("starting_agents", 0)
        ) / len(window)

    else:
        raise ValueError(f"{stage.name}: unknown success_metric {stage.success_metric!r}")

    return rate >= stage.success_threshold, rate


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
