"""
Evasion-rule variants for the rule-based heuristic, for cheap paired comparison on
stage 2 (single predator) via eval_baselines.py  (policy names "ev_<variant>").

Every variant keeps the heuristic's FORAGING behaviour untouched: when no predator is
in range the call goes straight to heuristic_policy.action_decision. Only what happens
when a predator is perceived differs. Nothing in src/ is modified.

Facts from src/ that the variants rely on
  * Observation angles are relative to the agent's facing (0 = ahead); a Predator entry
    has distance, angle, rel_dir. Edge entries carry 'coords' in the agent-local frame.
  * move_direction is applied relative to the PRE-turn facing; turn_angle is applied after.
  * Predator: hearing 60 (always-charge radius ~90 = hearing*1.5), vision 250; if the
    agent faces it and it is > ~90 away it flanks (slow) instead of charging; sprint 15
    vs agent sprint 20 (walking speed is lower than the predator's 11/15).

Variants (name -> knobs), react = ignore predators farther than this, sprint = sprint
when nearer than this (or when not facing it: it will charge regardless):
  base      react=inf sprint=85  (identical to the current heuristic; sanity check)
  zonesC    react=110 sprint=70   tight zones: forage on when the predator is far
  zones     react=140 sprint=85
  zonesB    react=180 sprint=110  early, generous zones
  wall      wall repulsion added to the escape vector (do not flee into obstacle edges)
  multi     escape vector = sum over perceived predators, weighted (100/d)^2
  face      keep facing the predator while backing away (turn toward it every tick)
  all       zones + wall + multi
"""
import math
from typing import Dict

from src.utils.DTOs import ActionRequest
from src.utils.controllers.heuristic_policy import (
    PREDATOR_FACING_CONE,
    _closest_point_on_edge,
    action_decision as base_action,
)

WALL_R = 70.0   # obstacle edges closer than this push the escape direction away from them
WALL_W = 1.5    # repulsion weight at contact (predator unit-vector weight is 1.0 at d=100)

INF = float("inf")

VARIANTS = {
    "base":   dict(react=INF, sprint=85.0, weighted=False, walls=False, face=False),
    "zonesC": dict(react=110.0, sprint=70.0, weighted=False, walls=False, face=False),
    "zones":  dict(react=140.0, sprint=85.0, weighted=False, walls=False, face=False),
    "zonesB": dict(react=180.0, sprint=110.0, weighted=False, walls=False, face=False),
    "wall":   dict(react=INF, sprint=85.0, weighted=False, walls=True, face=False),
    "multi":  dict(react=INF, sprint=85.0, weighted=True, walls=False, face=False),
    "face":   dict(react=INF, sprint=85.0, weighted=False, walls=False, face=True),
    "all":    dict(react=140.0, sprint=85.0, weighted=True, walls=True, face=False),
}


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def action(state: Dict, rng, name: str) -> ActionRequest:
    p = VARIANTS[name]
    obs = state["observations"]
    preds = [o for o in obs if o.get("type") == "Predator"]
    if not preds:
        return base_action(state, rng)

    near = min(preds, key=lambda o: o["distance"])
    d, a = near["distance"], near["angle"]

    if d > p["react"]:
        # Predator perceived but far: forage as if it were not there.
        st = dict(state)
        st["observations"] = [o for o in obs if o.get("type") != "Predator"]
        return base_action(st, rng)

    # ---- escape direction in the agent-local frame (0 = straight ahead) ----
    if p["weighted"]:
        vx = vy = 0.0
        for o in preds:
            if o["distance"] > p["react"]:
                continue
            w = (100.0 / max(o["distance"], 10.0)) ** 2
            vx += -math.cos(o["angle"]) * w
            vy += -math.sin(o["angle"]) * w
    else:
        vx, vy = -math.cos(a), -math.sin(a)

    if p["walls"]:
        for e in obs:
            if e.get("type") != "Edge":
                continue
            dist, ang = _closest_point_on_edge(e["coords"])
            if dist < WALL_R:
                w = (1.0 - dist / WALL_R) * WALL_W
                vx += -math.cos(ang) * w
                vy += -math.sin(ang) * w

    if math.hypot(vx, vy) < 1e-6:
        vx, vy = -math.cos(a), -math.sin(a)
    escape = math.atan2(vy, vx)

    sprint_speed = state["sprint_speed"]
    speed = state["speed"]
    charging = d < p["sprint"] or abs(a) > PREDATOR_FACING_CONE

    if charging:
        move_distance = sprint_speed
        turn_angle = a if p["face"] else escape
    else:
        move_distance = speed
        turn_angle = a * 0.6 if not p["face"] else a

    return ActionRequest(
        agent_id=state["agent_id"],
        move_distance=move_distance,
        move_direction=escape,
        turn_angle=turn_angle,
        spawn_agent=False,
    )
