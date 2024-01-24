import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback

# create the environment
env = gym.make('Pendulum-v1', render_mode='rgb_array', g=2)

# create the model
model = PPO('MlpPolicy', env, verbose=1)

# create weights and biases callback
wandb.init(project="pendulumRL", sync_tensorboard=True)
wandb_callback = WandbCallback(model_save_freq=10000,
                                model_save_path=f"models/{wandb.run.id}",
                                verbose=2,
                                )


# train the model in a loop and save the weights incrementally
# for i in range(100):
#     model.learn(total_timesteps=10000)
#     model.save(f'./pendulum_models/pendulum_{i}')

model.learn(total_timesteps=1000000, callback=wandb_callback)



# # test the environment
# obs, info = env.reset()

# for i in range(1000):
#     action = env.action_space.sample()
#     print(f'action: {action}')
#     obs, rewards, terminated, truncated, info  = env.step(action)
#     print(f'obs: {obs}, reward: {rewards}, terminated: {terminated}, truncated: {truncated}')
#     env.render()
