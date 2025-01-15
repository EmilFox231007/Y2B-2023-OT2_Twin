import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from ot2_gym_wrapper import OT2Env  # Custom environment wrapper
import wandb
from wandb.integration.sb3 import WandbCallback

# Set up the environment
env = OT2Env()
env = Monitor(env)
env = DummyVecEnv([lambda: env])

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
args = parser.parse_args()

# Set up Weights & Biases
wandb.init(project="Task11-RL", sync_tensorboard=True)
wandb.config.update({
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size,
    "n_steps": args.n_steps,
    "n_epochs": args.n_epochs
})

# Create a W&B callback for logging
wandb_callback = WandbCallback(
    model_save_freq=1000,
    model_save_path=f"models/{wandb.run.id}",
    verbose=2,
)

# Initialize the model
model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{wandb.run.id}")

# Total timesteps for training
total_timesteps = 100000

# Define a directory to save the models locally
model_dir = f"models/{wandb.run.id}"
os.makedirs(model_dir, exist_ok=True)

# Training loop
for i in range(10):
    # Train the model and save locally
    model.learn(total_timesteps=total_timesteps, callback=wandb_callback, reset_num_timesteps=False, tb_log_name=f"runs/{wandb.run.id}")
    
    # Save the model every 5 iterations or on the last iteration
    if (i + 1) % 5 == 0 or (i + 1) == 10:
        model.save(f"{model_dir}/model_{total_timesteps * (i + 1)}.zip")

wandb.finish()
print("Training completed!")
