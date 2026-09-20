"""
Reproduction-rule variants for the rule-based heuristic, for paired comparison on
stage 4/5 (reproduction enabled) via eval_baselines.py  (policy names "rp_<variant>").

Movement / foraging / evasion = the unchanged heuristic (heuristic_policy.action_decision).
Only the spawn flag is recomputed. Predator perceived -> never spawn (same as the heuristic).

IMPORTANT: a deployed agent cannot see the global population. Everything here uses only what
one agent perceives: n_near = number of OTHER agents in its observation (hearing radius +
vision cone), fruits within its hearing radius, its own energy fraction. So "population
below/above X" is approximated by LOCAL density. (The earlier heuristic_capN test used the
true global count, which is privileged, and still did not help.)

Baseline rule (heuristic): spawn if energy_fraction > 0.30 and a fruit is within hearing radius.
Note the environment itself refuses/charges: needs energy > 100 (0.20 of the default 500 max).

  e22      energy_fraction > 0.22 (barely above the 100-energy floor)  -> more births
  e40      energy_fraction > 0.40                                       -> fewer, better-fed births
  food2    require >= 2 fruits within hearing radius
  dens4    block spawning when >= 4 other agents perceived
  dens8    block spawning when >= 8 other agents perceived
  isol     isolated agent (0 others perceived): threshold 0.22, else baseline rule
  tier     the three-band rule:  n_near <= 1  -> baseline rule (avoid extinction)
                                 2 <= n_near <= 5 -> strict: energy_fraction > 0.50 and >= 2 fruits
                                 n_near >= 6 -> blocked
"""
from typing import Dict

from src.utils.DTOs import ActionRequest
from src.utils.controllers.heuristic_policy import (
    SPAWN_ENERGY_FRACTION,
    action_decision as base_action,
)

VARIANTS = ("e22", "e40", "food2", "dens4", "dens8", "isol", "tier")


def _spawn(state: Dict, name: str) -> bool:
    obs = state["observations"]
    if any(o.get("type") == "Predator" for o in obs):
        return False
    ef = state["energy"] / state["max_energy"] if state["max_energy"] > 0 else 0.0
    hr = state["hearing_radius"]
    n_fruit = sum(1 for o in obs if o.get("type") == "Fruit" and o["distance"] <= hr)
    n_near = sum(1 for o in obs if o.get("type") == "Agent")
    base = ef > SPAWN_ENERGY_FRACTION and n_fruit >= 1

    if name == "e22":
        return ef > 0.22 and n_fruit >= 1
    if name == "e40":
        return ef > 0.40 and n_fruit >= 1
    if name == "food2":
        return ef > SPAWN_ENERGY_FRACTION and n_fruit >= 2
    if name == "dens4":
        return base and n_near < 4
    if name == "dens8":
        return base and n_near < 8
    if name == "isol":
        if n_near == 0:
            return ef > 0.22 and n_fruit >= 1
        return base
    if name == "tier":
        if n_near <= 1:
            return base
        if n_near <= 5:
            return ef > 0.50 and n_fruit >= 2
        return False
    raise KeyError(name)


def action(state: Dict, rng, name: str) -> ActionRequest:
    a = base_action(state, rng)
    return ActionRequest(
        agent_id=a.agent_id,
        move_distance=a.move_distance,
        move_direction=a.move_direction,
        turn_angle=a.turn_angle,
        spawn_agent=_spawn(state, name),
    )
