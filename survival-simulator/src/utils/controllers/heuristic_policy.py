import math
import random
from typing import Dict, List, Optional, Tuple

from src.utils.DTOs import ActionRequest

"""
First model: hand-tuned heuristic / rule-based policy.

This encodes, as explicit priority rules, everything worked out in the
project's design analysis (energy budgeting, predator avoidance,
reproduction strategy, biome awareness) rather than learning anything.
It is meant to be a safe, fast, fully-understood baseline to validate the
whole pipeline (agent_server -> validation/evaluation) before attempting a
learned (RL) policy later.

Priority per agent, each tick:
  1. Predator threat -> evade (sprint away if a charge is imminent, or
     retreat while re-orienting toward the predator if it's still far and
     we're already generally facing it, to keep it in its slower "flanking"
     behavior instead of a direct sprint charge).
  2. No threat -> forage toward the nearest sensed fruit, or failing that
     the nearest sensed tree, walking (never sprinting for food - it isn't
     going anywhere, and sprinting is ~10x the energy cost of walking).
  3. No threat and nothing sensed -> gentle biased-random exploration
     (small turns, walking speed) instead of the dummy policy's fully
     random heading every tick.
  4. Independently of the movement decision: spawn a new agent only when
     no predator is present, energy is comfortably above the flat 100
     cost (> 50% of this agent's own max_energy), and the agent is
     currently near sensed food (so the child is born somewhere that can
     plausibly feed both agents afterward).

All the thresholds below are exactly the ones derived in the project's
"Energy Budgeting" / "Predator Avoidance" / "Reproduction Strategy" /
"Biome Awareness" notes.
"""

# A predator's hearing_radius is hardcoded to 60 in Predator.__init__ and is
# never overridden by spawn_predator(), so its "always charge, regardless of
# facing" radius (hearing_radius * 1.5) is reliably 90. We react a little
# before that line to allow for one tick of latency.
PREDATOR_CHARGE_DISTANCE = 85.0
# A predator only chooses the slower flanking approach when the target is
# generally facing it (within this half-angle). Outside of it, it always
# charges directly regardless of distance.
PREDATOR_FACING_CONE = math.pi / 2

# Reproduce only when energy is comfortably above the flat 100-energy cost,
# expressed as a fraction of this agent's own (possibly mutated) max_energy
# so it scales correctly regardless of trait mutations. Empirically (see
# the First Model notes), a foraging agent under this policy rarely climbs
# much past ~35-40% of max_energy in practice - a 0.5 threshold was
# measured to never fire at all, which silently disabled reproduction
# entirely and doomed the whole population once every individual aged out.
# 0.3 was chosen by testing against that measured ceiling, low enough to
# actually trigger, while the code still refuses to spawn under 100 energy
# regardless.
SPAWN_ENERGY_FRACTION = 0.3

# How close an obstacle edge has to be, roughly ahead of us, before we
# nudge sideways to avoid wasting movement energy walking into it. This is
# a light refinement only - the environment itself already deflects
# collisions automatically, this just avoids repeatedly paying for blocked
# movement attempts.
EDGE_AVOID_DISTANCE = 15.0
EDGE_AVOID_TURN = 0.4

# Amplitude of the random heading drift used while exploring blind (nothing
# sensed at all). Deliberately small and biased-continuous rather than a
# uniform random heading every tick, so the agent actually covers new
# ground instead of jittering in place like the dummy policy does.
EXPLORE_TURN_RANGE = 0.15


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _closest_point_on_edge(coords: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
    """
    Given an edge's two endpoints in the agent-local frame (already rotated
    so the agent is at the origin facing angle 0, exactly as environment.py
    hands them to us via the "Edge" observation), return (distance, angle)
    of the closest point on the segment to the agent. Same technique
    predator.py itself uses for its own wall-avoidance.
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


def action_decision(observation_response: Dict, rng: random.Random) -> ActionRequest:
    """
    Hand-tuned heuristic action selection for one agent.

    Args:
        observation_response (dict): This agent's ObservationResponse payload.
        rng (random.Random): Random number generator, used only for the
            gentle exploration wander when nothing is sensed.

    Returns:
        ActionRequest: Action decision for this agent this tick.
    """
    agent_id = observation_response["agent_id"]
    observations: List[Dict] = observation_response["observations"]
    energy = observation_response["energy"]
    max_energy = observation_response["max_energy"]
    speed = observation_response["speed"]
    sprint_speed = observation_response["sprint_speed"]

    energy_fraction = (energy / max_energy) if max_energy > 0 else 0.0

    predators = [o for o in observations if o.get("type") == "Predator"]
    fruits = [o for o in observations if o.get("type") == "Fruit"]
    trees = [o for o in observations if o.get("type") == "Tree"]
    edges = [o for o in observations if o.get("type") == "Edge"]

    nearest_predator: Optional[Dict] = min(predators, key=lambda o: o["distance"]) if predators else None

    # --- 1. Predator threat takes priority over everything else. ---
    if nearest_predator is not None:
        angle = nearest_predator["angle"]  # relative to our current facing, 0 = ahead
        distance = nearest_predator["distance"]

        # away_offset is relative to our CURRENT (pre-turn) facing, since
        # environment.agent_step applies this tick's movement using the old
        # direction, before applying turn_angle. Using it for move_direction
        # makes us physically retreat this tick regardless of which branch
        # we take below.
        away_offset = _wrap(angle + math.pi)

        charging = distance < PREDATOR_CHARGE_DISTANCE or abs(angle) > PREDATOR_FACING_CONE
        if charging:
            # Already within charge range, or not oriented toward it: the
            # predator is going to (or already does) sprint straight at us.
            # Our sprint_speed is faster than its by default (20 vs 15), so
            # a full commitment to fleeing directly away wins a straight
            # race, and this reframes into a short energy burst rather than
            # a sustained chase, since predators exhaust their own energy
            # after a few seconds of sprinting.
            turn_angle = away_offset
            move_direction = away_offset
            move_distance = sprint_speed
        else:
            # Still beyond the charge radius and we're already generally
            # facing it: reinforce that orientation (keeps the predator in
            # its slower flanking behavior instead of switching to a direct
            # charge) while still physically retreating at walking speed -
            # no need to burn sprint energy yet.
            turn_angle = angle * 0.6
            move_direction = away_offset
            move_distance = speed

        return ActionRequest(
            agent_id=agent_id,
            move_distance=move_distance,
            move_direction=move_direction,
            turn_angle=turn_angle,
            spawn_agent=False,
        )

    # --- 2. No threat: forage toward the nearest sensed food. ---
    nearest_fruit: Optional[Dict] = min(fruits, key=lambda o: o["distance"]) if fruits else None
    nearest_tree: Optional[Dict] = min(trees, key=lambda o: o["distance"]) if trees else None

    if nearest_fruit is not None and (nearest_tree is None or nearest_fruit["distance"] <= nearest_tree["distance"]):
        target = nearest_fruit
    else:
        target = nearest_tree

    if target is not None:
        target_angle = target["angle"]
        target_distance = target["distance"]
        # Cap the move to the target's actual distance so we don't overshoot
        # past it - the same technique predator.py itself uses
        # (min(sprint_speed, distance_to_agent)). Without this, once within
        # one tick's walking distance of a tree (which has no collision and
        # isn't itself edible) the agent overshoots past it, flips ~180
        # degrees to re-target it, and repeats indefinitely - burning huge
        # amounts of both turn and movement energy while eating nothing.
        # This was measured directly: it accounted for the majority of all
        # energy expenditure in the first (uncapped) version of this policy.
        if target_distance < 1.0:
            # Already essentially on top of it - hold heading to avoid
            # jitter from a near-degenerate angle calculation at ~0 distance.
            turn_angle = 0.0
            move_direction = 0.0
            move_distance = 0.0
        else:
            turn_angle = target_angle
            move_direction = target_angle
            # Walking, not sprinting: food doesn't flee, and sprinting costs
            # an order of magnitude more energy per unit distance than
            # walking.
            move_distance = min(speed, target_distance)
    else:
        # --- 3. Nothing sensed at all: gentle biased exploration. ---
        turn_angle = rng.uniform(-EXPLORE_TURN_RANGE, EXPLORE_TURN_RANGE)
        move_direction = 0.0
        move_distance = speed

    # Light wall-avoidance nudge: if an obstacle edge is close and roughly
    # ahead, bias away from it so we don't repeatedly pay movement energy
    # for a step the environment will just deflect anyway.
    if edges:
        edge_dist, edge_angle = min(
            (_closest_point_on_edge(e["coords"]) for e in edges),
            key=lambda t: t[0],
        )
        if edge_dist < EDGE_AVOID_DISTANCE and abs(edge_angle) < math.pi / 2:
            avoid = EDGE_AVOID_TURN if edge_angle < 0 else -EDGE_AVOID_TURN
            turn_angle += avoid
            move_direction += avoid

    # --- 4. Reproduction gate, independent of the movement chosen above. ---
    # Gate on an actual nearby FRUIT, not merely a visible tree (a tree
    # might not have any fruit yet). Measured directly: gating on
    # "fruit or tree visible" let the population overshoot to 50 agents
    # before mass starvation collapsed it back to zero shortly after -
    # classic boom-bust from reproducing on too weak a food signal.
    # Requiring the fruit to be within hearing_radius (i.e. actually near,
    # not just seen far off in the vision cone) is a stronger, more honest
    # "this spot can currently support another mouth to feed" signal.
    hearing_radius = observation_response["hearing_radius"]
    locally_fed = nearest_fruit is not None and nearest_fruit["distance"] <= hearing_radius
    spawn_agent = energy_fraction > SPAWN_ENERGY_FRACTION and locally_fed

    return ActionRequest(
        agent_id=agent_id,
        move_distance=move_distance,
        move_direction=move_direction,
        turn_angle=turn_angle,
        spawn_agent=spawn_agent,
    )
