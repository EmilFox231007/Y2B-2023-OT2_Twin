from sim_class import Simulation
import os
import numpy as np
import time
import cv2

# set current directory to working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# num robots
num_agents = 1
# Create the simulation
sim = Simulation(num_agents, render=False, rgb_array=True)

steps = 100000
n = 100
start = time.time()
for step in range(steps):
    actions = [[-0.1,0.1,0.1,0]]*num_agents
    # drop every 60th step
    if step % 60 == 0:
        actions = [[0,0,0,1]]*num_agents
    
    obs = sim.run(actions)
    #print(sim.current_frame)
    # display the current frame in a window using OpenCV
    #print(sim.current_frame.shape)

    cv2.imshow('frame', np.array(sim.current_frame))
    cv2.waitKey(1)
    if step % n == 0:
        fps = n / (time.time() - start)
        start = time.time()
        print(f'fps: {fps}')
        #print(obs)
        #print(sim.droplet_positions)

    # reset after 1000 steps
    if step == 1000:
        sim.reset(num_agents)
    