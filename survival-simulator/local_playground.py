import os
import pygame
import random
from src.core import SimulationCore
from src.utils.controllers.heuristic_policy import action_decision as heuristic_action_decision

# Same toggle as agent_server.py: off by default, so this still shows the
# heuristic unless you explicitly ask for the RL policy. See
# training/RL/inference.py's module docstring for the current checkpoint's
# known performance gap before relying on this for anything but watching it.
USE_RL_POLICY = os.environ.get("USE_RL_POLICY", "0") == "1"

if USE_RL_POLICY:
    from training.RL.inference import action_decision as rl_action_decision

def local_simulation(verbose=True):
    seed = None
    if seed is None: # If no seed is provided, generate a random one
        seed = random.randint(0, 2**32 - 1)

    sim = SimulationCore(seed=seed)
    action_rng = random.Random(seed) # Deterministic actions. Can be removed if action_decision is deterministic
    policy = rl_action_decision if USE_RL_POLICY else heuristic_action_decision

    pygame.init()
    screen, clock = None, None

    # Optional render
    if verbose:
        info = pygame.display.Info()
        env_ratio = sim.env_width / sim.env_height
        screen_height = int(info.current_h * 0.9) # 90% of screen height
        screen_width = int(screen_height * env_ratio) # Keep aspect ratio
        screen = pygame.display.set_mode((screen_width, int(screen_height)), pygame.SCALED)
        clock = pygame.time.Clock()
        pygame.display.set_caption(f"Survival Simulator -- {'RL' if USE_RL_POLICY else 'heuristic'} policy")

    running = True
    actions = []

    while running:
        if verbose:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Check if user closes window
                    running = False

        state = sim.step(actions)

        actions = []
        for agent, agent_state in zip(sim.env.agents, state["observations"]):
            action = policy(agent_state, action_rng)
            actions.append((agent.agent_id, action))

        if verbose:
            sim.env.draw(screen)
            font = pygame.font.SysFont(None, 24)
            img = font.render(f'Score: {state["score"]:.2f}', True, (255,255,255))
            screen.blit(img, (20, 20))
            pygame.display.flip()
            clock.tick(60) # Control max FPS

        print(f'Score: {state["score"]:.2f} | Agents alive: {state["num_agents"]:.0f} | Time: {sim.env.time:.2f}')

        if state["num_agents"] == 0 or sim.env.time > 3000:
            print(f"Game over! Final Score: {state['score']}")
            print(f"Seed: {seed}")
            running = False

    pygame.quit()

if __name__ == "__main__":
    local_simulation(verbose=True)
