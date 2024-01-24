from stable_baselines3 import PPO
from ot2_gym_wrapper import CustomEnv
import numpy as np

num_agents = 1

# Load the trained agent
model = PPO.load("ppo_pipette copy")

# Test the trained agent (example)
env = CustomEnv(num_agents, render=True)
obs, info = env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info  = env.step(action)
    # calculate the distance between the pipette and the goal
    distance = obs[3:] - obs[:3]
    # calculate the error between the pipette and the goal
    error = np.linalg.norm(distance)

    print(f'goal: {obs[3:]}, pipette: {obs[:3]}, reward: {rewards}, terminated: {terminated}, truncated: {truncated}, error: {error}')
    
    # change the goal if the pipette is within 0.013
    if error < 0.009:
        obs, info = env.reset()

    if terminated:
        obs, info = env.reset()