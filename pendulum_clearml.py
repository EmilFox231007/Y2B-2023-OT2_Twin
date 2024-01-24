from stable_baselines3 import PPO
import gymnasium as gym
import time
from clearml import Task

# Replace Pendulum-v1/YourName with your own project name (Folder/YourName, e.g. 2022-Y2B-RoboSuite/Michael)
task = Task.init(project_name='Pendulum-v1/Dean', task_name='Experiment1')
#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/robosuite:py3.8-2')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")

env = gym.make('Pendulum-v1',g=9.81,render_mode='rgb_array')

model = PPO('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=10000, progress_bar=True)


# test_env = gym.make('Pendulum-v1',g=9.81,render_mode='human')
# #Test the trained model
# obs, info = env.reset()
# for i in range(1000):
#     action, _ = model.predict(obs,deterministic=True)
#     obs, reward, done, terminated, info = env.step(action)
#     env.render()
#     time.sleep(0.025)
#     if done:
#         env.reset()