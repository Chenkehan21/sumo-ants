import torch
import torch.nn as nn
import torch.optim as optim

import gym
from gym import spaces
import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter
import random


HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9
PATH = "./CEM_FrozenLake.pth"


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(OneHotWrapper, self).__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Discrete)
        obs_shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_shape
        )
    
    def _observation(self, obs):
        onehot_obs = np.copy(self.observation_space.low)
        onehot_obs[obs] = 1.0
        return onehot_obs


class Net(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Net, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(input_shape, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_shape),
        )
    
    def forward(self, x):
        return self.pipe(x)


Episodes = namedtuple("Episodes", field_names=["Rewards", "Steps"])
EpisodeSteps = namedtuple("EpisodeSteps", field_names=["Observation", "Action"])


def generate_batch(env, net):
    sm = nn.Softmax(dim=1)
    obs = env.reset()
    episode_rewards = 0.0
    batch = []
    episod_steps = []
    while True:
        obs_v = torch.FloatTensor([obs])
        actions = net(obs_v)
        actions_probability = sm(actions).data.numpy()[0]
        action = np.random.choice(len(actions_probability), p=actions_probability)
        next_obs, reward, done, _ = env.step(action)
        episode_rewards += reward
        step = EpisodeSteps(Observation=obs, Action=action) # very important! don't store tensor(obs)
        episod_steps.append(step)
        if done:
            episode = Episodes(Rewards=episode_rewards, Steps=episod_steps)
            batch.append(episode)
            episode_rewards = 0.0
            obs = env.reset()
            episod_steps = []  
            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []
        else:
            obs = next_obs


def filter_batch(batch):
    filter_fun = lambda x: x.Rewards * (GAMMA ** len(x.Steps))
    discounted_rewards = list(map(filter_fun, batch))
    reward_bound = np.percentile(discounted_rewards, PERCENTILE)

    obs = []
    action_labels = []
    elite_batch = []
    for examples, disc_rewards in zip(batch, discounted_rewards):
        if disc_rewards > reward_bound:
            obs.extend(map(lambda x: x.Observation, examples.Steps))
            action_labels.extend(map(lambda x: x.Action, examples.Steps))
            elite_batch.append(examples)

    return obs, action_labels, elite_batch, reward_bound


def train(env, net, criterion, optimizer):
    print("start training")
    net.train()
    writer = SummaryWriter(comment="CEM_FL")

    train_batch = []
    for iter_n, batch in enumerate(generate_batch(env, net)):
        mean_reward = float(np.mean(list(map(lambda x: x.Rewards, batch)))) # men_reward cant't calculate with train_batch
        obs, action_labels, train_batch, reward_bound = filter_batch(train_batch + batch)
        if not train_batch:
            continue
        obs_v = torch.FloatTensor(obs)
        action_labels_v = torch.LongTensor(action_labels)
        train_batch = train_batch[-500:]

        optimizer.zero_grad()
        action = net(obs_v)
        loss = criterion(action, action_labels_v)
        loss.backward()
        optimizer.step()
        print("iter %d: loss=%5.3f, mean_reward=%5.3f, reward_bound=%5.3f, batch=%5.3f" % (
            iter_n, loss, mean_reward, reward_bound, len(train_batch)))
        writer.add_scalar("loss_CEM_FL", loss, iter_n)
        writer.add_scalar("mean_reward_CEM_FL", mean_reward, iter_n)
        writer.add_scalar("reward_bound_CEM_FL", reward_bound, iter_n)

        if iter_n > 15000:
            print("Done!")
            break
    writer.close()
    torch.save(net.state_dict(), PATH)


def test(env, net):
    print("start testing!")
    env = gym.wrappers.Monitor(env, "recording")
    net.eval()
    net.load_state_dict(torch.load(PATH))
    total_reward = 0.0

    with torch.no_grad():
        obs = env.reset()
        while True:
            obs = torch.FloatTensor([obs])
            _, action = torch.max(net(obs), dim=1)
            action = action.item()
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                print("total reward=%.3f" % total_reward)
                break
            else:
                obs = next_obs
    env.close()
    env.env.close()
    gym.upload("/home/chenkehan/RESEARCH/codes/try/DL_RL/recording_FL")


def main(to_train=True, to_test=True):
    random.seed(1)
    env = OneHotWrapper(gym.make("FrozenLake-v0"))
    input_shape = env.observation_space.shape[0]
    output_shape = env.action_space.n
    net = Net(input_shape, output_shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    
    if to_train:
        train(env, net, criterion, optimizer)
    if to_test:
        test(env, net)


if __name__ == "__main__":
    # main(to_train=True, to_test=False)
    env = OneHotWrapper(gym.make("FrozenLake-v0"))