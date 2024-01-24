import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_agents, render=False):
        super(CustomEnv, self).__init__()
        self.simulation = Simulation(num_agents, render)
        self.num_agents = num_agents

        # Define action and observation space
        # These should be gym.spaces objects
        # Example: spaces.Box, spaces.Discrete, etc.

        # The action space is a 3x1 vector of floats [x_velocity, y_velocity, z_velocity] from -1 to 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # The observation space is a 6x1 vector of floats [pipette_x_position, pipette_y_position, pipette_z_position, goal_x_position, goal_y_position, goal_z_position]
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # Define a random goal position within the workspace
        # x_range: [-0.187, 0.253]
        # y_range: [-0.1705, 0.2195]
        # z_range: [0.1695, 0.2896]
        self.goal_position = np.array([np.random.uniform(-0.187, 0.253), np.random.uniform(-0.1705, 0.2195), np.random.uniform(0.1695, 0.2896)])

        # set the start position of the pipette to a random position within the workspace
        self.simulation.set_start_position(np.random.uniform(-0.16, 0.2), np.random.uniform(-0.16, 0.2), np.random.uniform(0.14, 0.25))

        self.steps = 0
        self.min_error = 1000000

    def step(self, action):
        # Execute one time step within the environment
        # append a 0 to the action to account for the drop action
        action = np.append(action, 0)
        # Modify as per your simulation's step logic
        observation = self.simulation.run([action])
        # extract the pipette position from the observation
        observation = observation[f'robotId_{self.simulation.robotIds[0]}']['pipette_position']
        # Add the goal position to the observation
        observation = np.concatenate((observation, self.goal_position)).astype(np.float32)
        # convert the observation to a numpy array
        observation = np.array(observation).astype(np.float32)
        # Calculate the reward: the inverse of the distance between the pipette and the goal
        #reward = 1 / (np.linalg.norm(observation[:3] - observation[3:])+0.0001)
        error = np.linalg.norm(observation[:3] - observation[3:])
        if error < 0.995*self.min_error:
            self.min_error = error
            reward = 1
        else:
            reward = 0

        # define the done condition if the pipette is within 0.01 of the goal
        if np.linalg.norm(observation[:3] - observation[3:]) < 0.001:
            terminated = True
            reward += 1000
            #observation, _ = self.reset()
        else:
            terminated = False
        #truncate the episode if it has been running for 1000 steps
        if self.steps > 1500:
            truncated = True
            reward -= 10
            #observation, _ = self.reset()
        else:
            truncated = False
        #truncated = False
        info = {}  # Additional info for debugging, if needed
        self.steps += 1
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        # set the goal position
        self.goal_position = np.array([np.random.uniform(-0.187, 0.253), np.random.uniform(-0.1705, 0.2195), np.random.uniform(0.1695, 0.2896)])
        # Optionally, handle the seed if you need reproducibility
        if seed is not None:
            np.random.seed(seed)
        # Reset the state of the environment to an initial state
        self.simulation.reset(self.num_agents)
        observation = self.simulation.get_states()
        # extract the pipette position from the observation
        observation = observation[f'robotId_{self.simulation.robotIds[0]}']['pipette_position']
        # Add the goal position to the observation
        observation = np.concatenate((observation, self.goal_position)).astype(np.float32)
        # convert the observation to a numpy array
        observation = np.array(observation).astype(np.float32)
        self.steps = 0
        self.min_error = 1000000

        #set the start position of the pipette to a random position within the workspace
        self.simulation.set_start_position(np.random.uniform(-0.16, 0.2), np.random.uniform(-0.16, 0.2), np.random.uniform(0.14, 0.25))

        return observation, {}

    def render(self, mode='human', close=False):
        # Render the environment if needed
        # This can be as simple or complex as needed for your simulation
        pass

    def close(self):
        self.simulation.close()

