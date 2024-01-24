import gymnasium as gym
from ot2_gym_wrapper import CustomEnv

env = CustomEnv(1)
env.reset()

while True:
    action = env.action_space.sample()
    action[0][3] = 0
    obs = env.step(action)