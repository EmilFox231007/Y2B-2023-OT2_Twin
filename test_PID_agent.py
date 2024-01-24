#script to create a PID agent and test it in the gym environment
from ot2_gym_wrapper import CustomEnv
import numpy as np
from simple_pid import PID
import time

# create the PID controllers for each axis
pid_x = PID(30, 10, 0.5, setpoint=0)
pid_y = PID(30, 10, 0.5, setpoint=0)
pid_z = PID(30, 10, 0.5, setpoint=0)

num_agents = 1
# Test the trained agent (example)
env = CustomEnv(num_agents, render=True)
obs, info = env.reset()
plant_no = 0
goal_positions = np.array([[0.18275-0.15/2, 0.163-0.15/2, 0.18]])
print(goal_positions)
env.goal_position= goal_positions[plant_no]
#set the setpoint for each axis
pid_x.setpoint = env.goal_position[0]
pid_y.setpoint = env.goal_position[1]
pid_z.setpoint = env.goal_position[2]
time.sleep(1)
for i in range(1000000):
    # get the current pipette position
    pipette_position = obs[:3]
    # get the current goal position
    goal_position = obs[3:]
    # calculate the distance between the pipette and the goal
    distance = goal_position - pipette_position
    # calculate the error between the pipette and the goal
    error = np.linalg.norm(distance)
    print(f'error: {error}')
    # calculate the PID output for each axis
    pid_x_output = pid_x(pipette_position[0])
    pid_y_output = pid_y(pipette_position[1])
    pid_z_output = pid_z(pipette_position[2])
    # create the action array
    action = np.array([pid_x_output, pid_y_output, pid_z_output, 0])

    obs, rewards, terminated, truncated, info  = env.step(action)

    print(f'goal: {obs[3:]}, pipette: {obs[:3]}, reward: {rewards}, terminated: {terminated}, truncated: {truncated}, error: {error}')
    
    # change the goal if the pipette is within the required error
    if error <= 0.001:
        print('dropping')
        print('dropping')
        print('dropping')
        time.sleep(1)
        action = np.array([0, 0, 0, 1])
        obs, rewards, terminated, truncated, info  = env.step(action)
        print('dropped')
        # get the current pipette position
        pipette_position = obs[:3]
        # get the current goal position
        goal_position = obs[3:]
        # calculate the distance between the pipette and the goal
        distance = goal_position - pipette_position
        # calculate the error between the pipette and the goal
        error = np.linalg.norm(distance)
        plant_no += 1
        #obs, info = env.reset()
        if plant_no > len(goal_positions)-1:
            env.goal_position = np.array([-0, 0, 0.18])
            #set the setpoint for each axis
            pid_x.setpoint = env.goal_position[0]
            pid_y.setpoint = env.goal_position[1]
            pid_z.setpoint = env.goal_position[2]
            print('moved to end')
        else:
            env.goal_position= goal_positions[plant_no]
            #set the setpoint for each axis
            pid_x.setpoint = env.goal_position[0]
            pid_y.setpoint = env.goal_position[1]
            pid_z.setpoint = env.goal_position[2]
            print('moved to next plant')
            del error

    # if terminated:
    #     obs, info = env.reset()
    #     print('terminated')