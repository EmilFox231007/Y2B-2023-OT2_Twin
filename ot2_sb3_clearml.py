from stable_baselines3 import PPO, A2C
from ot2_gym_wrapper import CustomEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

from clearml import Task

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='Test/Dean', # NB: Replace YourName with your own name
                    task_name='Dean_Experiment1')

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

#add weights and biases logging
#import wandb
# run = wandb.init(project="ot2_sb3", sync_tensorboard=True)

#from wandb.integration.sb3 import WandbCallback

# wandb_callback = WandbCallback(model_save_freq=10000,
#                                 model_save_path=f"models/{run.id}",
#                                 verbose=2,
#                                 )

num_agents = 1
# Create your custom environment
env = CustomEnv(num_agents, render=False)
# It will check your custom environment and output additional warnings if needed
#check_env(env)
#env = Monitor(env, "./monitor_logs", allow_early_resets=True)

# Initialize an SB3 agent
model = PPO("MlpPolicy", 
            env, 
            verbose=1, 
            n_steps= 2048*4, 
            seed=42, 
            batch_size=64,
            n_epochs=10,)
            #tensorboard_log=f"runs/{run.id}")
#model = A2C("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=2350000, progress_bar=True)#, callback=wandb_callback)

#close the environment
env.close()

# # Save the agent
# model.save("ppo_pipette")

# del model # remove to demonstrate saving and loading

# # Load the trained agent
# model = PPO.load("ppo_pipette")

# # Test the trained agent (example)
# env = CustomEnv(num_agents, render=True)
# obs, info = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, terminated, truncated, info  = env.step(action)
#     env.render()
