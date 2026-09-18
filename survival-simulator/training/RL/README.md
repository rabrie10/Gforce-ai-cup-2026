# RL/

Placeholder -- the RL policy work (custom training environment, curriculum
learning stages, PPO/MAPPO training, saved actor checkpoints, and the code
that loads a trained model into agent_server.py at evaluation time) lives
here once we get to it. Deliberately empty for now: the plan is to build and
sweep the rule-based baseline in ../rule_based/ first, since its results
(what actually kills the colony, and the real per-step throughput) are what
the curriculum and the "is RL worth it here" decision should be based on,
not guessed in advance.
