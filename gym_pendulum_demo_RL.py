import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO

# Create the environment
env = gym.make('Pendulum-v1', render_mode='rgb_array', g=2)

# Create the model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model in a loop and save the weights incrementally
for i in range(100):
    model.learn(total_timesteps=10000)
    model.save(f'./pendulum_models/pendulum_{i}')


# # Reset the environment
# observation, info = env.reset()
# # test the environment
# for i in range(1000):
#     action = env.action_space.sample()
#     print(f'action: {action}')
#     observation, reward, terminated, truncated, info  = env.step(action)
#     print(f'obs: {observation}, reward: {reward}, terminated: {terminated}, truncated: {truncated}')
#     env.render()

