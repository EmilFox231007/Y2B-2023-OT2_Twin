from sim_class import Simulation
import os
import numpy as np
import time
import cProfile
import pstats

# set current directory to working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# num robots
num_agents = 100
# Create the simulation
sim = Simulation(num_agents)

steps = 1000
n = 100
start = time.time()

# Create a Profile object
profiler = cProfile.Profile()

for step in range(steps):
    actions = [[-0.1,0.1,0.1,0]]*num_agents
    # drop every 60th step
    if step % 60 == 0:
        actions = [[0,0,0,1]]*num_agents

    # Start profiling for the 'run' method
    profiler.enable()
    obs = sim.run(actions)
    profiler.disable()

    if step % n == 0:
        fps = n / (time.time() - start)
        start = time.time()
        print(f'fps: {fps}')

    # reset after 1000 steps
    if step == 1000:
        sim.reset(num_agents)

# Print out the profiling statistics
profiler.print_stats(sort='time')
profiler.dump_stats("profile_results.prof")

