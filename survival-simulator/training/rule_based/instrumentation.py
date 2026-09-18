"""
Instrumentation for the rule-based baseline sweep.

Everything here attaches to an already-constructed src.elements.environment.Environment
instance from the OUTSIDE, by rebinding two of its methods on that one instance
(monkeypatching). Nothing in src/ is read or modified by this file at runtime beyond
calling the environment's own public methods.

Why the death-cause classification is exact, not a guess
----------------------------------------------------------
`kill_agent()` is called from exactly two places in src/elements/environment.py's
`non_agent_step`:

  1. The agent loop, when `agent.energy <= 0` (starvation). This branch always
     `continue`s immediately after killing, so a starved agent never reaches the
     predator loop this tick.
  2. The predator loop, which only searches `self.agents` / the spatial grid for
     targets -- and any agent that starved earlier in the SAME tick has already
     been removed from both by step 1's call to kill_agent (which rebuilds the
     grid). So the predator loop can only ever kill an agent whose energy is
     still positive.

Because those two call sites are mutually exclusive by construction, checking
`agent.energy <= 0` vs `> 0` at the moment kill_agent is invoked deterministically
tells you which path triggered it -- not an inferred heuristic.

There is no separate "died of old age" branch in the source: `age > max_age` just
adds extra energy drain that makes the same starvation branch fire sooner. We
still tag it, as a post-hoc split of the starvation cause using the agent's age
relative to its max_age at the moment of death, since that's the only information
available to distinguish "ran out of food while still young" from "aging finally
caught up with it."

Why the score decomposition doesn't need to hook fruit-eating at all
----------------------------------------------------------------------
Environment.score is one float with three additive terms per tick:

    score += dt                              (always, unconditionally)
    score += fruit.energy / 1000             (per fruit eaten, inline in the agent loop)
    score -= agent.energy / 100              (per agent eaten, inline in the predator loop)

The predation penalty is captured exactly by the kill_agent hook above (we read
agent.energy right there, before it's touched by anything else). The dt term is a
known constant. So for any tick:

    fruit_bonus_this_tick = score_delta_this_tick - dt + predation_penalty_this_tick

This is computed in run_episode.py (it needs access to the per-tick death list
sliced to "since the last tick", which lives at the run-loop level, not here).
"""
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List

_biome_render_patched = False


def patch_environment_for_headless() -> None:
    """
    Class-level patch (not per-instance, since this must take effect before an
    Environment even exists): replaces Environment._render_biome_surface with a
    no-op for the rest of this process.

    That method is a plain Python double for-loop over every pixel of the map
    (width * height -- 1,920,000 iterations at the default 1600x1200), calling
    rng.choice() and surface.set_at() on each one, purely to paint the biome
    onto a pygame Surface for on-screen drawing. Nothing about the simulation
    reads that surface: self.biome_map (the actual per-tile gameplay data --
    move_penalty, energy_drain_rate, tree/fruit spawn rates) is built earlier
    by Map_generator.generate() and is untouched by this patch. Headless sweeps
    never call env.draw(), so that surface is pure wasted work here -- and a
    real cost: with sweep.py's multiprocessing, several worker processes each
    hit this ~1.9M-iteration pure-Python loop at once for every new seed's
    environment, which is enough sustained CPU load across cores to make the
    whole machine (not just the sweep) sluggish or unresponsive.

    Safe to call multiple times (e.g. once per worker process) -- idempotent.
    Only affects Environment objects created AFTER this is called.
    """
    global _biome_render_patched
    if _biome_render_patched:
        return
    from src.elements.environment import Environment

    def _skip_render(self):
        pass

    Environment._render_biome_surface = _skip_render
    _biome_render_patched = True


@dataclass
class RunLog:
    """Per-episode instrumentation state, populated purely by the hooks below plus
    per-tick sampling done by record_tick()."""
    ticks: List[Dict[str, Any]] = field(default_factory=list)
    deaths: List[Dict[str, Any]] = field(default_factory=list)
    births: List[Dict[str, Any]] = field(default_factory=list)


def attach_instrumentation(env, log: RunLog) -> None:
    """
    Rebind env.kill_agent and env.spawn_agent on this ONE Environment instance
    to record death/birth events into `log`. Call once, right after the
    Environment (or SimulationCore) is constructed, before the step loop starts.
    """
    original_kill_agent = env.kill_agent
    original_spawn_agent = env.spawn_agent

    def kill_agent_hook(self, agent):
        # Must classify BEFORE calling the original: kill_agent removes the
        # agent from self.agents / self.agents_dict, but never touches
        # agent.energy / agent.age, so reading them here is still safe either way.
        if agent.energy <= 0:
            cause = "starvation_old" if agent.age > agent.max_age else "starvation_young"
        else:
            cause = "predator"
        log.deaths.append({
            "sim_time": self.time,
            "agent_id": agent.agent_id,
            "cause": cause,
            "energy": agent.energy,
            "age": agent.age,
            "max_age": agent.max_age,
            "max_energy": agent.max_energy,
        })
        return original_kill_agent(agent)

    def spawn_agent_hook(self, x=None, y=None, parent=None):
        child = original_spawn_agent(x=x, y=y, parent=parent)
        if child is not None and parent is not None:
            log.births.append({
                "sim_time": self.time,
                "agent_id": child.agent_id,
                "parent_id": parent.agent_id,
                "speed": child.speed,
                "sprint_speed": child.sprint_speed,
                "max_energy": child.max_energy,
                "hearing_radius": child.hearing_radius,
                "vision_radius": child.vision_radius,
                "cone_angle": child.cone_angle,
            })
        return child

    env.kill_agent = types.MethodType(kill_agent_hook, env)
    env.spawn_agent = types.MethodType(spawn_agent_hook, env)


def record_tick(env, log: RunLog, prev_score: float, predation_penalty: float, dt: float) -> float:
    """
    Call once per simulation tick, right after env has been stepped, to sample
    population/energy state and decompose this tick's score delta into its three
    additive terms. `predation_penalty` is the sum of agent.energy/100 for every
    predator-caused death recorded THIS tick (the caller slices log.deaths itself,
    since only the run loop knows where "this tick" starts in that list).

    Returns the new prev_score, to pass back in on the next call.
    """
    score_delta = env.score - prev_score
    fruit_bonus = score_delta - dt + predation_penalty

    energies = [a.energy for a in env.agents]
    ages = [a.age for a in env.agents]

    log.ticks.append({
        "sim_time": env.time,
        "score": env.score,
        "score_delta": score_delta,
        "time_bonus": dt,
        "fruit_bonus": fruit_bonus,
        "predation_penalty": predation_penalty,
        "num_agents": len(env.agents),
        "num_predators": len(env.predators),
        "num_fruits": len(env.fruits),
        "num_trees": len(env.trees),
        "mean_energy": (sum(energies) / len(energies)) if energies else 0.0,
        "median_energy": (sorted(energies)[len(energies) // 2] if energies else 0.0),
        "mean_age": (sum(ages) / len(ages)) if ages else 0.0,
    })
    return env.score
