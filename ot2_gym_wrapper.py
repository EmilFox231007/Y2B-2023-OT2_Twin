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
        self.goal_space = spaces.Box(low=np.array([-0.1904, -0.1712, -0.1205]), high=np.array([0.255, 0.2203, 0.2906]), shape=(3,), dtype=np.float32)
        
        # Keep track of steps and previous distance
        self.steps = 0
        self.prev_distance = None

    def seed(self, seed=None):
        """Ensure reproducibility."""
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self, seed=None):
        """Reset the environment and get new goal."""
        self.seed(seed)
        self.goal_position = self.goal_space.sample()
        goal_position = np.array(self.goal_position, dtype=np.float32)

        # Reset simulation and get initial pipette position
        observation = self.sim.reset(num_agents=1)
        robot_id = list(observation.keys())[0]
        pipette_position = np.array(self.sim.get_pipette_position(int(robot_id.replace("robotId_", ""))), dtype=np.float32)

        # Combine pipette position and goal position
        observation = np.concatenate([pipette_position, goal_position])

        self.steps = 0
        self.prev_distance = np.linalg.norm(pipette_position - goal_position)  # Initialize distance tracking
        return observation, {}

    def step(self, action):
        """Execute one time step within the environment."""
        action = np.append(action, 0)  # Append 0 for drop action
        scaled_action = np.clip(action * np.array([0.5, 0.5, 0.3]), -1.0, 1.0)  # Scale per axis
        
        # Call simulation step
        observation = self.sim.run([scaled_action])

        # Extract pipette position
        robot_id = list(observation.keys())[0]
        pipette_position = np.array(self.sim.get_pipette_position(int(robot_id.replace("robotId_", ""))), dtype=np.float32)
        goal_position = np.array(self.goal_position, dtype=np.float32)

        # Compute new observation
        observation = np.concatenate([pipette_position, goal_position])

        # Compute reward
        reward = self.calculate_reward(pipette_position, goal_position)

        # Check if task is complete
        terminated = np.linalg.norm(pipette_position - goal_position) < self.get_threshold()
        truncated = self.steps >= self.max_steps or (self.steps > 50 and abs(self.prev_distance - np.linalg.norm(pipette_position - goal_position)) < 0.001)
        
        # Log every 10 steps
        if self.steps % 10 == 0:
            print(f"Step {self.steps}: Pos={pipette_position}, Goal={goal_position}, Reward={reward:.2f}, Distance={np.linalg.norm(pipette_position - goal_position):.4f}")

        self.steps += 1
        return observation, reward, terminated, truncated, {}

    def calculate_reward(self, pipette_position, goal_position):
        """Dynamically scaling reward function."""
        current_distance = np.linalg.norm(pipette_position - goal_position)
        base_penalty = -10 * current_distance  # Higher penalty for being far
        progress_reward = 5 * (self.prev_distance - current_distance) if self.prev_distance is not None else 0
        goal_bonus = 100 if current_distance < self.get_threshold() else 0
        self.prev_distance = current_distance
        return float(base_penalty + progress_reward + goal_bonus)

    def get_threshold(self):
        """Adaptive threshold based on workspace size."""
        return 0.002 * np.mean([0.255 - (-0.1904), 0.2203 - (-0.1712), 0.2909 - 0.1205])

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
