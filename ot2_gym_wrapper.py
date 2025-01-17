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
        reward = -distance * 10  # Scale the distance to make the penalty more significant
        if distance < self.threshold:
            reward += 100  # Bonus for completion
        return reward

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.goal_position = self.goal_space.sample()
        observation = self.sim.reset(num_agents=1)
        pipette_position = observation['pipette_position']
        
        return np.concatenate([pipette_position, self.goal_position])

    def step(self, action):
        # Add exploration noise
        action += np.random.normal(0, 0.1, size=self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        observation = self.sim.run(action)
        pipette_position = observation['pipette_position']
        reward = self.calculate_reward(pipette_position, self.goal_position)
        distance = np.linalg.norm(pipette_position - self.goal_position)
        done = distance < self.threshold or self.steps >= self.max_steps
        self.steps += 1
        
        return np.concatenate([pipette_position, self.goal_position]), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
