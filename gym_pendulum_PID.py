import gymnasium as gym
from simple_pid import PID
from math import atan2, degrees, pi

# Create the PID controller
pid_controller = PID(Kp=10.0, Ki=0.1, Kd=1)
pid_controller.setpoint = 0

# Create the pendulum environment
env = gym.make('Pendulum-v1', render_mode='human', g=2)

# Reset the environment
observation, info = env.reset()

done = False
while not done:
    # determine the angle of the pendulum from the observation - [x, y, angular velocity]
    angle = atan2(observation[1], observation[0])
    print(f'angle: {degrees(angle)}')
    # Get the action from the PID controller
    action = pid_controller(angle)
    # Perform the action
    observation, reward, truncated, done, info = env.step([action])
    # Render the environment
    env.render()
