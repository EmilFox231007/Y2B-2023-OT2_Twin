import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback
import wandb
from ot2_gym_wrapper import OT2Env  # Custom environment wrapper
from clearml import Task  # Import ClearML's Task
import typing_extensions
import numpy as np

# Initialize the environment
env = OT2Env(render=False)
env = Monitor(env)  # Wrap the environment to monitor the rewards

# Initialize ClearML Task
task = Task.init(project_name="Mentor Group S/Group 3", task_name="RL_train_group_3")
task.set_base_docker("deanis/2023y2b-rl:latest")
task.execute_remotely(queue_name="default")

os.environ['WANDB_API_KEY'] = 'ad7961aa16de00343e0cd159062e4c79502f7184'

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
args = parser.parse_args()

# Initialize wandb
run = wandb.init(project="Task11-RL", sync_tensorboard=True)
wandb_callback = WandbCallback(model_save_freq=1000, model_save_path=f"models/{run.id}", verbose=2)

model = PPO('MlpPolicy', env, verbose=1, learning_rate=args.learning_rate, 
            batch_size=args.batch_size, n_steps=args.n_steps, n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}")

time_steps = 500000
save_interval = 5
no_improvement_intervals = 3  # Number of intervals to wait without improvement
best_reward = float('-inf')
patience_counter = 0  # Counter for early stopping

try:
    for i in range(10):
        model.learn(total_timesteps=time_steps, callback=wandb_callback, reset_num_timesteps=False, tb_log_name=f"runs/{run.id}")
        mean_reward = np.mean([ep_info['r'] for ep_info in env.get_episode_rewards()])
        wandb.log({"Mean Reward": mean_reward})

        # Check for improvement
        if mean_reward > best_reward:
            best_reward = mean_reward
            patience_counter = 0
            model.save(f"models/{run.id}/best_model")
        else:
            patience_counter += 1
        
        # Early stopping condition
        if patience_counter >= no_improvement_intervals:
            print("Stopping early due to no improvement")
            break

        if (i + 1) % save_interval == 0 or (i + 1) == 10:
            model.save(f"models/{run.id}/{time_steps * (i + 1)}")
except Exception as e:
    print(f"An error occurred: {e}")
