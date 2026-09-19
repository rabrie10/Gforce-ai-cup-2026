"""
Loads a trained ActorCritic checkpoint and serves ActionRequest objects
straight from raw ObservationResponse-shaped dicts -- same call signature
as src/utils/controllers/heuristic_policy.py's action_decision(agent_state,
rng), so agent_server.py can import either one interchangeably (see the
USE_RL_POLICY toggle wired into agent_server.py).

Runs the network in eval() mode with deterministic=True (uses the
tanh-squashed mean action directly, no sampling): at evaluation time we
want the policy's best guess, not exploration noise, which is only useful
during training.

CURRENT STATUS (be aware before flipping the toggle on for a real
submission): the stage5_full_dynamics checkpoint plateaued at ~100%
extinction rate and ~120-155s mean survival out of a 3000s max during
training -- see training/RL/README.md and the per-stage episode CSVs in
training/RL/checkpoints/. The existing rule-based heuristic
(src/utils/controllers/heuristic_policy.py) averaged ~512s to extinction
over 30 seeds. As of this checkpoint, the RL policy is meaningfully WORSE
than the heuristic, not a drop-in upgrade. Don't submit it as the active
policy until a newer checkpoint actually beats the heuristic on a real
comparison (see the bottom of this file for a quick way to check).
"""
import os
import sys
from typing import Dict, Optional

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.DTOs import ActionRequest
from training.RL.env_wrapper import OBS_DIM, decode_action, encode_observation
from training.RL.network import ACTION_DIM, ActorCritic

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "stage5_full_dynamics_final.pt")

_net: Optional[ActorCritic] = None
_net_path: Optional[str] = None


def get_policy(checkpoint_path: Optional[str] = None) -> ActorCritic:
    """
    Loads and caches the network. Re-loads only if a DIFFERENT checkpoint
    path is requested than what's currently cached (useful for swapping
    checkpoints without restarting the server, e.g. in a quick comparison
    script) -- normal server usage just calls this with no argument every
    time and gets the cached instance back.
    """
    global _net, _net_path
    path = checkpoint_path or os.environ.get("RL_CHECKPOINT", DEFAULT_CHECKPOINT)
    if _net is None or _net_path != path:
        net = ActorCritic(OBS_DIM, ACTION_DIM)
        state_dict = torch.load(path, map_location="cpu")
        net.load_state_dict(state_dict)
        net.eval()
        _net = net
        _net_path = path
    return _net


@torch.no_grad()
def action_decision(agent_state: Dict, rng=None, checkpoint_path: Optional[str] = None) -> ActionRequest:
    """
    Same call signature as heuristic_policy.action_decision(agent_state,
    rng) -- rng is accepted but unused, since the trained policy is run
    deterministically (mean action, no sampling) at evaluation time.
    """
    net = get_policy(checkpoint_path)
    obs = encode_observation(agent_state)
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    action_clipped, _, _, _ = net.act(obs_t, deterministic=True)
    action_np = action_clipped[0].numpy()
    return decode_action(agent_state["agent_id"], action_np, agent_state["sprint_speed"])


if __name__ == "__main__":
    # Quick, no-server sanity check + a real side-by-side comparison against
    # the heuristic, using the exact same instrumented episode runner the
    # rule_based sweep uses (so the numbers are directly comparable to the
    # ~512s heuristic baseline already on record).
    #
    #   python3 training/RL/inference.py --seed 1 --max-sim-time 600
    import argparse
    import random as pyrandom

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from src.core import SimulationCore
    from training.rule_based.instrumentation import patch_environment_for_headless

    parser = argparse.ArgumentParser(description="Run one headless episode with the trained RL policy and print the outcome.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-sim-time", type=float, default=600.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    patch_environment_for_headless()
    sim = SimulationCore(seed=args.seed, dt=1 / 10)
    actions = []
    while sim.env.time <= args.max_sim_time and len(sim.env.agents) > 0:
        state = sim.step(actions)
        actions = [
            (agent.agent_id, action_decision(agent_state, None, args.checkpoint))
            for agent, agent_state in zip(sim.env.agents, state["observations"])
        ]
    extinct = len(sim.env.agents) == 0
    print(f"seed={args.seed}  extinction_time={sim.env.time if extinct else 'survived to cap'}  "
          f"final_score={sim.env.score:.2f}  final_agents={len(sim.env.agents)}")
