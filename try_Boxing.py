import gym

env = gym.make("Boxing-v0")
obs = env.reset()
total_reward = 0.0
while True:
    env.render()
    # action = env.action_space.sample()
    action = 10
    next_obs, reward, done, _ = env.step(action)
    if done:
        print("total reward: %d" % total_reward)
        break
    obs = next_obs
env.close()