import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import pybullet as p

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Define goal space
        goal_low = np.array([-0.1904, -0.1712, -0.1205], dtype=np.float32)
        goal_high = np.array([0.255, 0.2203, 0.2906], dtype=np.float32)
        self.goal_space = spaces.Box(low=goal_low, high=goal_high, dtype=np.float32)

        self.steps = 0
        self.prev_distance = None  # To track improvement
        self.goal_position = np.zeros(3, dtype=np.float32)  # Ensure goal_position is always an array

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Set a random goal position
        self.goal_position = np.array(self.goal_space.sample(), dtype=np.float32)  # Explicitly convert to NumPy array

        # Reset simulation and get initial pipette position
        observation = self.sim.reset(num_agents=1)
        robot_id = list(observation.keys())[0]
        robot_id_str = robot_id.replace("robotId_", "")
        pipette_position = np.array(self.sim.get_pipette_position(int(robot_id_str)), dtype=np.float32)

        # Debug prints
        print(f"goal_position type: {type(self.goal_position)}, value: {self.goal_position}")
        print(f"pipette_position type: {type(pipette_position)}, value: {pipette_position}")

        # Combine pipette position and goal position
        observation = np.concatenate([pipette_position, self.goal_position])

        self.steps = 0
        self.prev_distance = np.linalg.norm(pipette_position - self.goal_position)  # Initialize distance tracking
        return observation, {}

    def step(self, action):
        # Append zero for the drop action
        action = np.append(action, 0)

        # Run the simulation with the given action
        observation = self.sim.run([action])

        # Process observation
        robot_id = list(observation.keys())[0]
        robot_id_str = robot_id.replace("robotId_", "")
        pipette_position = np.array(self.sim.get_pipette_position(int(robot_id_str)), dtype=np.float32)

        # Construct new observation
        observation = np.concatenate([pipette_position, self.goal_position])

        # Compute reward using the calculate_reward function
        reward = self.calculate_reward(pipette_position, self.goal_position)

        # Check if task is complete
        terminated = np.linalg.norm(pipette_position - self.goal_position) < self.get_threshold()
        truncated = self.steps >= self.max_steps  # Check if max steps reached

        self.steps += 1
        return observation, reward, terminated, truncated, {}

    def calculate_reward(self, pipette_position, goal_position):
        """Computes a dynamically scaling reward based on distance to the goal."""
        current_distance = np.linalg.norm(pipette_position - goal_position)

        # Scale the penalty to be stronger when further away, weaker when closer
        base_penalty = -10 * current_distance  

        # If the pipette moves closer than before, reward progress
        if self.prev_distance is not None:
            progress_reward = 5 * (self.prev_distance - current_distance)  # Reward improvement
        else:
            progress_reward = 0  

        # Bonus reward for reaching goal
        goal_bonus = 100 if current_distance < self.get_threshold() else 0

        # Save the current distance for the next step
        self.prev_distance = current_distance

        # Final reward calculation
        return float(base_penalty + progress_reward + goal_bonus)

    def get_threshold(self):
        """Computes a reasonable threshold based on workspace size."""
        x_range = 0.255 - (-0.1904)
        y_range = 0.2203 - (-0.1712)
        z_range = 0.2909 - 0.1205
        return 0.001 * (x_range + y_range + z_range) / 3  # 0.1% of average range

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
