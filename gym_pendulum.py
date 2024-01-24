import gymnasium as gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# Create the environment
env = gym.make('Pendulum-v1')
env = Monitor(env, "./monitor_logs", allow_early_resets=True)

# Initialize the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=1000000)

# Test the trained agent
test_env = gym.make('Pendulum-v1', render_mode="human")
obs, info = test_env.reset()
print(obs)
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, trunc, info  = test_env.step(action)
    print(obs)
    env.render()

# create the pendulum
# env = gym.make('Pendulum-v1')
# env.reset()
# while True:
#     action = env.action_space.sample()
#     obs, rewards, dones, info = env.step(action)
#     time.sleep(0.1)
#     env.render()
