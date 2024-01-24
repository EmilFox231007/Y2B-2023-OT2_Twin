from stable_baselines3 import PPO
from ot2_gym_wrapper import CustomEnv
import numpy as np
import time

num_agents = 1

# Load the trained agent
model = PPO.load("ppo_pipette copy")

# Test the trained agent (example)
env = CustomEnv(num_agents, render=True)
obs, info = env.reset()
plant_no = 0
goal_positions = np.array([[0.154, 0.21, 0.18],[0.158, 0.18, 0.18],[0.153, 0.155, 0.18],[0.157, 0.125, 0.18],[0.154, 0.095, 0.18]])
env.goal_position= goal_positions[plant_no]
time.sleep(1)
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info  = env.step(action)
    # calculate the distance between the pipette and the goal
    distance = obs[3:] - obs[:3]
    # calculate the error between the pipette and the goal
    error = np.linalg.norm(distance)

    print(f'goal: {obs[3:]}, pipette: {obs[:3]}, reward: {rewards}, terminated: {terminated}, truncated: {truncated}, error: {error}')
    
    # change the goal if the pipette is within the required error
    if error < 0.016:#0.00066:
        action = np.array([0, 0, 0, 1])
        obs, rewards, terminated, truncated, info  = env.step(action)
        plant_no += 1
        #obs, info = env.reset()
        if plant_no > 4:
            env.goal_position = np.array([-0.5, 0.5, 0.18])
        else:
            env.goal_position= goal_positions[plant_no]

    if terminated:
        obs, info = env.reset()