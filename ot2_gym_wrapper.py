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
        self.sim = Simulation(num_agents=1)
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        goal_low = np.array([-0.1904, -0.1712, -0.1205], dtype=np.float32)
        goal_high = np.array([0.255, 0.2203, 0.2906], dtype=np.float32)
        self.goal_space = spaces.Box(low=goal_low, high=goal_high, dtype=np.float32)
        self.threshold = 0.001
        self.steps = 0

    def calculate_reward(self, pipette_position, goal_position):
        distance = np.linalg.norm(pipette_position - goal_position)
        reward = -10 * distance
        if distance < self.threshold:
            reward += 100
        return reward

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.goal_position = np.array(self.goal_space.sample(), dtype=np.float32)  # Ensure goal is an array
        observation = self.sim.reset(num_agents=1)
        robot_id = list(observation.keys())[0]
        robot_id_str = robot_id.replace("robotId_", "")
        pipette_position = np.array(self.sim.get_pipette_position(int(robot_id_str)), dtype=np.float32)  # Ensure array
        observation = np.concatenate([pipette_position, self.goal_position])
        self.steps = 0
        info = {'message': 'Reset successful'}
        return observation, info


    def step(self, action):
        action = np.append(action, [0])
        observation = self.sim.run([action])
        robot_id = list(observation.keys())[0]
        robot_id_str = robot_id.replace("robotId_", "")
        pipette_position = np.array(self.sim.get_pipette_position(int(robot_id_str)), dtype=np.float32)  # Ensure array
        goal_position = np.array(self.goal_position, dtype=np.float32)  # Ensure array
        observation = np.concatenate([pipette_position, goal_position])
        reward = self.calculate_reward(pipette_position, goal_position)
        distance = np.linalg.norm(pipette_position - goal_position)
        terminated = distance < self.threshold
        truncated = self.steps >= self.max_steps
        reward -= 1  # Step penalty
        self.steps += 1
        return observation, reward, terminated, truncated, {}


    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
