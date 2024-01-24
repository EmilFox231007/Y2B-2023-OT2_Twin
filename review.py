from sim_class import Simulation

sim = Simulation(1, render=True, rgb_array=True)

while True:
    actions = [[-0.5,0.5,0.5,1]]
    obs = sim.run(actions)
    print(obs)
