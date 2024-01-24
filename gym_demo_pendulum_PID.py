# Creating a PID Controller to solve the Pendulum-v1 environment
import gymnasium as gym
from simple_pid import PID
from math import atan2, degrees, pi

# Create the PID controller
pid_controller = PID(Kp=6, Ki=0.8, Kd=4)
pid_controller.setpoint = pi/2 #90 degrees = pi/2 radians

# Create the pendulum environment
env = gym.make('Pendulum-v1', render_mode='human', g=2)

# Reset the environment
observation, info = env.reset()

for i in range(1000):
    # determine the angle of the pendulum from the observation - [x, y, angular velocity]
    angle = atan2(observation[1], observation[0])
    print(f'angle: {degrees(angle)}')
    # Get the action from the PID controller
    action = pid_controller(angle)
    # Perform the action
    observation, reward, terminated, truncated, info = env.step([action])
    # Render the environment
    env.render()
