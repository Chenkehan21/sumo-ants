from lib import wrappers, dqn_model

import torch
import gym
import numpy as np
import argparse


DEFAULT_ENV_NAME = "Pong-v0"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME, help="Environment to play")
    parser.add_argument("-r", "--record", help="directory for video")
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda str, _:stg)
    net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0

    while True:
        state_v = torch.tensor(np.array([state]), copy=False)
        q_values = net(state_v).data.numpy()[0]
        action = np.argmax(q_values)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        state = next_state
    print("total reward: %.3f" % total_reward)

    if args.record:
        env.env.close()