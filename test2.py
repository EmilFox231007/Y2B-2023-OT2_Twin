from sim_class import Simulation
import os
import numpy as np
import time

# set current directory to working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# num robots
num_agents = 1
# Create the simulation
sim = Simulation(num_agents)

steps = 1000
n = 100
start = time.time()
for step in range(steps):
    actions = [[-0.1,0.1,0.1,0]]*num_agents
    obs = sim.run(actions)
    if step % n == 0:
        fps = n / (time.time() - start)
        start = time.time()
        print(f'fps: {fps}')
        print(obs)
