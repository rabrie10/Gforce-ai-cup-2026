import os
import random
from fastapi import FastAPI, Body
from src.utils.DTOs import StepResponse
from src.utils.controllers.heuristic_policy import action_decision as heuristic_action_decision

# Off by default: as of the current checkpoint (stage5_full_dynamics_final),
# the trained RL policy plateaued at ~100% extinction and ~120-155s mean
# survival in training -- meaningfully worse than the heuristic's ~512s
# average (see training/RL/inference.py's module docstring and
# training/RL/README.md). Set USE_RL_POLICY=1 only once a checkpoint has
# actually been verified to beat the heuristic (training/RL/inference.py
# run standalone gives a quick single-episode check; compare against
# training/rule_based/results/sweep_summary.csv for the real baseline).
USE_RL_POLICY = os.environ.get("USE_RL_POLICY", "0") == "1"

if USE_RL_POLICY:
    from training.RL.inference import action_decision as rl_action_decision

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI(title="Survival Simulator Agent Endpoint")

@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    """
    Receives the current simulation state and returns actions for all agents.
    """
    rng = random.Random(1)  # deterministic for testing
    policy = rl_action_decision if USE_RL_POLICY else heuristic_action_decision
    actions = [policy(agent.dict(), rng).dict() for agent in step.agent_status]

    # Must return {"actions": [...]} format
    return {"actions": actions}

@app.get("/")
def index():
    return {"message": "Agent endpoint running!", "policy": "rl" if USE_RL_POLICY else "heuristic"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
