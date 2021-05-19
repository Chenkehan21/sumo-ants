import gym
import numpy as np


class NewObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(NewObservationWrapper, self).__init__(env)
        self.action = np.random.sample((1, 3))
        self.init_action = np.array((0., 0., 0.))

    def reset(self):
        obs = self.env.reset()
        obs = np.concatenate((obs, self.init_action), axis=None)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, self.action), axis=None)
        return obs, reward, done, info
    

if __name__ == "__main__":
    env = NewObservationWrapper(gym.make("CartPole-v0"))
    obs = env.reset()
    total_reward = 0.0

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            print(obs)
            print(f"Got total rewards: {total_reward}")
            break
