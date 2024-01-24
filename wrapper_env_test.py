from ot2_gym_wrapper import CustomEnv

num_agents = 1

# Test the trained agent (example)
env = CustomEnv(num_agents, render=True)
obs, info = env.reset()
for i in range(1000):
    #action = env.action_space.sample()
    action = [-0.5,-0.5,0.8]
    obs, rewards, terminated, truncated, info  = env.step(action)
    print(f'goal: {obs[3:]}, pipette: {obs[:3]}, reward: {rewards}, terminated: {terminated}, truncated: {truncated}')