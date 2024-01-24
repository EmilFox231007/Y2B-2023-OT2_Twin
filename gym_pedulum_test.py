import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from math import atan2, degrees

# create the environment
env = gym.make('Pendulum-v1', render_mode='human', g=2)

# load the model
model = PPO.load('./pendulum_models/pendulum_10')

# test the environment
obs, info = env.reset()

for i in range(1000):
    action = model.predict(obs, deterministic=True)[0]
    #print(f'action: {action}')
    angle = atan2(obs[1], obs[0])
    print(f'angle: {degrees(angle)}')
    obs, rewards, terminated, truncated, info  = env.step(action)
    #print(f'obs: {obs}, reward: {rewards}, terminated: {terminated}, truncated: {truncated}')
    env.render()