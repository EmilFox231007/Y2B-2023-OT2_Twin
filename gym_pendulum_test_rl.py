#Load in the PPO weights and test
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO

# Create the environment
env = gym.make('Pendulum-v1', render_mode='human', g=2)

# Load the model
model = PPO.load('./pendulum_models/pendulum_13')

# Test the environment
obs, info = env.reset()

for i in range(1000):
    action = model.predict(obs, deterministic=True)[0]
    obs, rewards, terminated, truncated, info  = env.step(action)
    #print(f'obs: {obs}, reward: {rewards}, terminated: {terminated}, truncated: {truncated}')
    env.render()