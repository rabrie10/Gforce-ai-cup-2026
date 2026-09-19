"""
Diagnostic hook for the "blocked reproduction still costs 100 energy" issue.

src/elements/environment.py's Environment.agent_step() does, after moving/turning:

    if spawn_agent and agent.energy > 100:
        self.spawn_agent(parent=agent)
        agent.energy -= 100

It never checks whether a child was actually created. In stages 1-3 curriculum.py
blocks births (spawn_agent(parent=...) returns None), so a policy that requests a
spawn while above 100 energy pays 100 energy for nothing.

hook_spawn(env, counts, mask) rebinds agent_step on ONE Environment instance (same
monkeypatch pattern as the rest of training/RL, never touching src/):

  * counts["wasted"] is incremented for every spawn request that WOULD BE / WAS
    charged with no child created:
      - mask=False: exact -- the request cost >= 99 energy and the agent count
        did not increase.
      - mask=True : the request is suppressed (spawn_agent forced to False, so no
        charge happens); counted when the agent is still above 100 energy after
        its move/turn, i.e. exactly the condition under which the real charge
        would have fired.
  * mask=True additionally removes the charge, i.e. simulates the fixed behaviour.

Only meaningful on stages with reproduction disabled (stages 1-3). With
reproduction enabled the spawn succeeds (agent count increases) and is not counted.
"""
import types


def hook_spawn(env, counts, mask):
    orig = env.agent_step
    counts.setdefault("wasted", 0)

    def hook(self, agent_id, move_distance, move_direction, turn_angle, spawn_agent=False):
        ag = self.agents_dict.get(agent_id)
        n_before = len(self.agents)
        e_before = ag.energy if ag is not None else 0.0
        if mask:
            orig(agent_id, move_distance, move_direction, turn_angle, False)
            if spawn_agent and ag is not None and ag.energy > 100:
                counts["wasted"] += 1
        else:
            orig(agent_id, move_distance, move_direction, turn_angle, spawn_agent)
            if spawn_agent and ag is not None and len(self.agents) == n_before and (e_before - ag.energy) >= 99.0:
                counts["wasted"] += 1

    env.agent_step = types.MethodType(hook, env)
