"""
Hybrid-controller variants (policy names "hy_<variant>" in eval_baselines.py).

Scope, given what the interface allows (checked in agent_server.py / creature.py):
  * The server gets ALL agent states in one call, but there are NO absolute positions.
    Agents do see each other with an id, distance and angle, and fruits with distance and angle.
    So "coordinated foraging" is implemented as a local, stateless rule: an agent skips a fruit
    if a perceived neighbour is clearly closer to that fruit than it is (geometry in the
    agent-local frame; no shared memory needed, works identically in the server).
  * "Memory-based evasion": per-agent memory of the last predator distance (module dict keyed by
    agent_id). Closing speed -> time-to-contact. Predators that are far AND not closing are
    ignored (agent keeps foraging, saves sprint energy); fast-closing ones trigger the
    unchanged heuristic (which already faces + backs away).
  * Not included, because measured as noise/harmful: population cap/soft control, trait-aware
    parent selection (tier/dens/e22/e40 tests all p=1.0), wall-repulsion (ev_wall/ev_all were
    significantly worse).

Variants
  co       yield fruits to a clearly closer neighbour (only if energy fraction > CO_MIN_EF)
  tti      time-to-contact filter on predators
  co_tti   both
  co2      like co, but larger yield margin (needs neighbour 25% closer)
"""
import math
from typing import Dict

from src.utils.DTOs import ActionRequest
from src.utils.controllers.heuristic_policy import action_decision as base_action

VARIANTS = ("co", "co2", "tti", "co_tti")

CO_MIN_EF = 0.35       # a hungry agent never yields
TTI_IGNORE_D = 110.0   # only consider ignoring predators beyond this distance
TTI_SAFE = 4.0         # seconds; predator further than this in time-to-contact is ignored
DT = 0.1

_last: Dict[int, tuple] = {}   # agent_id -> (age, predator_distance)


def _yield_filter(state: Dict, margin: float):
    obs = state["observations"]
    ef = state["energy"] / state["max_energy"] if state["max_energy"] > 0 else 0.0
    if ef <= CO_MIN_EF:
        return obs
    agents = [o for o in obs if o.get("type") == "Agent"]
    if not agents:
        return obs
    out = []
    for o in obs:
        if o.get("type") != "Fruit":
            out.append(o)
            continue
        fx, fy = o["distance"] * math.cos(o["angle"]), o["distance"] * math.sin(o["angle"])
        mine = o["distance"]
        yielded = False
        for g in agents:
            gx, gy = g["distance"] * math.cos(g["angle"]), g["distance"] * math.sin(g["angle"])
            if math.hypot(gx - fx, gy - fy) < mine * margin:
                yielded = True
                break
        if not yielded:
            out.append(o)
    return out


def _tti_filter(state: Dict, obs):
    preds = [o for o in obs if o.get("type") == "Predator"]
    aid, age = state["agent_id"], state["age"]
    if not preds:
        _last.pop(aid, None)
        return obs
    near = min(preds, key=lambda o: o["distance"])
    d = near["distance"]
    prev = _last.get(aid)
    _last[aid] = (age, d)
    if prev is None or age < prev[0] or d <= TTI_IGNORE_D:
        return obs
    closing = (prev[1] - d) / DT          # units per second, >0 = approaching
    tti = d / closing if closing > 1e-6 else float("inf")
    if tti > TTI_SAFE:
        return [o for o in obs if o.get("type") != "Predator"]
    return obs


def action(state: Dict, rng, name: str) -> ActionRequest:
    obs = state["observations"]
    if name in ("co", "co_tti"):
        obs = _yield_filter(state, 0.5)
    elif name == "co2":
        obs = _yield_filter(state, 0.75)
    if name in ("tti", "co_tti"):
        obs = _tti_filter(state, obs)
    st = dict(state)
    st["observations"] = obs
    a = base_action(st, rng)
    return a
