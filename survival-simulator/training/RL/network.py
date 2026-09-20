"""
Actor-critic network shared by every agent (IPPO: independent PPO with
parameter sharing -- one network, every agent's observation is just another
row in the batch). See the "NN based RL solution" Notion page for why IPPO
over MAPPO/attention for now: agents already perceive nearby agents/
predators locally via their own observation slots, and a centralized critic
would need a permutation-invariant encoder over a variable-size (1-50+,
changing every tick) joint state -- that's attention-architecture work, not
a small increment, so it's deferred rather than built into v1.

Continuous 4-D action space (see env_wrapper.decode_action): the mean is
tanh-squashed into [-1, 1] (bounded, since the raw ActorCritic head output
is unbounded), std is a learnable state-independent parameter (simpler and
usually enough at this scale; can be made state-dependent later if the
policy needs different exploration noise in different situations). Sampled
raw actions are clipped to [-1, 1] before being handed to decode_action --
log-probs are computed against the pre-clip Normal, which is the standard
practical shortcut (no tanh-squash log-density correction): simpler, and
fine here since clipping only bites at the tails of an already-narrow
Gaussian once std shrinks during training.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal

ACTION_DIM = 4
LOG_STD_MIN = -5.0   # std >= 0.0067 (a -3.0 floor was binding: cloned policies need std <~0.05 and PPO must be free to shrink it)
LOG_STD_MAX = 0.0    # std <= 1.0 (was e^1: heading noise swamped the signal)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = ACTION_DIM, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)  # std starts around exp(-0.5) ~ 0.6
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mean = torch.tanh(self.actor_mean(h))
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        """
        obs: (N, obs_dim). Returns:
          action_clipped: (N, action_dim) -- what actually gets decoded and sent to the env
          action_raw:     (N, action_dim) -- pre-clip sample, stored for the PPO update
          logprob:        (N,)            -- log-prob of action_raw under the sampling policy
          value:           (N,)            -- critic estimate for this obs
        """
        mean, std, value = self.forward(obs)
        if deterministic:
            action_raw = mean
        else:
            action_raw = Normal(mean, std).rsample()
        logprob = Normal(mean, std).log_prob(action_raw).sum(-1)
        action_clipped = torch.clamp(action_raw, -1.0, 1.0)
        return action_clipped, action_raw, logprob, value

    def evaluate_actions(self, obs: torch.Tensor, action_raw: torch.Tensor):
        """Used during the PPO update: recompute log-prob/entropy/value under the CURRENT params."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        logprob = dist.log_prob(action_raw).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy, value


GLOBAL_DIM = 8


class CentralCritic(nn.Module):
    """
    Centralized value function for CTDE training (training-time only).

    Input = the agent's own encoded observation + GLOBAL_DIM colony-level features
    (see env_wrapper.CurriculumEnv.global_features: population, mean/min energy
    fraction, fruit/tree/predator counts, elapsed time fraction, mean age). Those
    features are privileged -- a deployed agent cannot see them -- which is exactly
    why they only feed the critic. The actor (ActorCritic) never sees them, so the
    deployed policy and inference.py are unchanged.
    """

    def __init__(self, obs_dim: int, global_dim: int = GLOBAL_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
