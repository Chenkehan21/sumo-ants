import gym
import time

env = gym.make("Pong-v0")
env.reset()

for _ in range(60):
    obs, reward, done, _ = env.step(0)
    if done:
        break
    env.render()
env.close()