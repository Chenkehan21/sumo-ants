import torch
import torch.nn as nn
import torch.optim as optim

import gym
import numpy as np
from tensorboardX import SummaryWriter
from collections import namedtuple
import argparse


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
LEARNING_RATE = 0.01
PATH = './CROSSENTROPY_CartPole.pth'


'''
The core of the cross-entropy method is to throw away bad episodes and train on better ones. 
So, the steps of the method are as follows:
1. Play N number of episodes using our current model and environment. 
2. Calculate the total reward for every episode and decide on a reward boundary. 
   Usually, we use some percentile of all rewards, such as 50th or 70th.
3. Throw away all episodes with a reward below the boundary.
4. Train on the remaining "elite" episodes using observations as the input and issued actions as the desired output.
5. Repeat from step 1 until we become satisfied with the result.
'''


class Net(nn.Module):
    def __init__(self, input_shpe, hidden_size, actions_n):
        super(Net, self).__init__()
        self.pipe = nn.Sequential(
            nn.Linear(input_shpe, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, actions_n),
        )

    def forward(self, x):
        return self.pipe(x)


Episode = namedtuple("Episode", field_names=["rewards", "steps"])
EpisodeSteps = namedtuple("EpisodeSteps", field_names=["observation", "action"])


"""Play N number of episodes(BATCH_SIZE) using our current model and environment.
"""
def generate_batch(env, net, device):
    obs = env.reset()
    episode_rewards = 0.0
    sm = nn.Softmax(dim=1).to(device)
    episode_steps = []
    batch = []
    while True:
        obs_v = torch.FloatTensor([obs]).to(device) # need to add one dimension for batchsize!       
        # print(obs_v) 
        actions = net(obs_v)
        action_prob = sm(actions).to("cpu").data.numpy()[0] # important!
        action = np.random.choice(len(action_prob), p=action_prob)
        next_obs, reward, done, _ = env.step(action)
        episode_rewards += reward
        step = EpisodeSteps(observation=obs, action=action)
        episode_steps.append(step)

        if done:
            episode = Episode(rewards=episode_rewards, steps=episode_steps)
            batch.append(episode)
            episode_rewards = 0.0
            obs = env.reset()
            episode_steps = [] # can't use episode_steps.clear() because it will clear a list in-place then in Episode steps=[]
            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []
        else:
            obs = next_obs


def batch_filter(batch):
    rewards = list(map(lambda tmp: tmp.rewards, batch))

    """Calculate the total reward for every episode and decide on a reward boundary. 
    Usually, we use some percentile of all rewards, such as 50th or 70th.
    """
    reward_bound = np.percentile(rewards, PERCENTILE)
    mean_reward = float(np.mean(rewards)) # it was numpy.float we'd better turn it to float for SummaryWriter

    action_labels, obs = [], []
    for data in batch:
        reward, step = data.rewards, data.steps

        """Throw away all episodes with a reward below the boundary.
        """
        if reward < reward_bound:
            continue
        action_labels.extend(map(lambda s: s.action, step))
        obs.extend(map(lambda s: s.observation, step))

    # remember to convert labels to tensors
    action_labels = torch.LongTensor(action_labels)
    obs = torch.FloatTensor(obs)
    return obs, action_labels, mean_reward, reward_bound


def train(env, net, criterion, optimizer, device):
    print("start training")
    net.train()
    writer = SummaryWriter(comment="cartpole_CE")
    

    for iter_n, batch in enumerate(generate_batch(env, net, device)):
        """Train on the remaining "elite" episodes using observations as the input and issued actions as the desired output
        """
        obs, action_labels, mean_reward, reward_bound = batch_filter(batch)
        action = net(obs).to(device)
        action_labels = action_labels.to(device)
        optimizer.zero_grad()
        loss = criterion(action, action_labels)
        loss.backward()
        optimizer.step()
        print("loss:%5.3f, reward_mean:%5.3f, reward_bound:%5.3f" % (loss.item(), mean_reward, reward_bound))
        writer.add_scalar("loss", loss.item(), iter_n)
        writer.add_scalar("reward_mean", mean_reward, iter_n)
        writer.add_scalar("reward_bound", reward_bound, iter_n)
        
        """Repeat from step 1 until we become satisfied with the result
        """
        if mean_reward > 199:
            print("solved!")
            break
    writer.close()
    torch.save(net.state_dict(), PATH)


def test(env, net, device):
    print("start testing")
    env = gym.wrappers.Monitor(env, "recording")
    net.eval()
    net.load_state_dict(torch.load(PATH))
    total_reward = 0.0

    with torch.no_grad():
        obs = env.reset()
        while True:
            obs = torch.FloatTensor([obs]).to(device)
            _, action = torch.max(net(obs).to("cpu"), dim=1)
            action = action.item()
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                print("total reward: %.2f" % total_reward)
                break
            else:
                obs = next_obs
    env.close()
    env.env.close()
    gym.upload("/home/chenkehan/RESEARCH/codes/try/DL_RL/recording")


def main(to_train=True, to_test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable Cuda Computation")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v0")
    input_shape = env.observation_space.shape[0]
    action_n = env.action_space.n
    net = Net(input_shape, HIDDEN_SIZE, action_n).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

    if to_train:
        train(env, net, criterion, optimizer, device)
    if to_test:
        test(env, net, device)


if __name__ == "__main__":
    main(to_train=False, to_test=True)