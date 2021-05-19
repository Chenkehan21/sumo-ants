import gym
from gym import spaces
import random
import numpy as np


class Env:
    def __init__(self):
        self.steps_left = 20
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(10, 5, 3))
    
    def reset(self):
        return np.random.random((10, 5, 3))


class Agent(Env):
    def __init__(self):
        super(Agent, self).__init__()
        self.total_reward = 0.0

    def step(self, action):
        obs = self._get_obs()
        self.steps_left -= 1
        reward = action * np.random.sample()
        self.total_reward += reward
        done = (self.steps_left == 0)
        return obs, reward, done

    def _get_obs(self):
        return np.random.random((10, 5, 3))


if __name__ == "__main__":
    env = Env()
    agent = Agent()
    env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done = agent.step(action)
        if done:
            print(f"the agent goes {env.steps_left} steps and get total reward: {agent.total_reward}")
            break

