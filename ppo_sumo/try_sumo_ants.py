import gym
import gym_compete
from policy import LSTMPolicy
import tensorflow as tf
import pickle
import numpy as np
import argparse
from tensorboardX import SummaryWriter
import time
import os

from lib import model, test_net, calc_logprob
from myutils import experience
import ptan

import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = "sumo-ants-v0"
GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2094
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

TEST_ITERS = 10


path = "/home/chenkehan/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl"

def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params


def setFromFlat(var_list, flat_params):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})


def calc_adv_ref(trajectory, net_crt, states_v, device="cpu"):
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]),
                                     reversed(values[1:]),
                                     reversed(trajectory[:-1])): # steps_count must equal to 1 !!
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val) # Q = A + V

    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)


"""My agent is agent1 and trained AI is agent0, now we want to train a normal ant fighter
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="enable cuda")

    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(comment="sumo-ants-ppo")
    save_path = "/home/chenkehan/RESEARCH/codes/try/DL_RL/ppo_sumo_ants/save_train_data"

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    obs_shape = env.observation_space.spaces[1].shape[0]
    action_shape = env.action_space.spaces[1].shape[0]
    net_act = model.ModelActor(obs_shape, action_shape).to(device)
    net_crt = model.ModelCritic(obs_shape).to(device)
    print(net_act)
    print(net_crt)

    agent = model.AgentA2C(net_act, device=device)
    exp_source = experience.MAExperienceSource(env, agent, steps_count=1)

    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    trajectory = []
    best_reward = None

    for step_idx, exp in enumerate(exp_source):
        reward_steps = exp_source.pop_rewards_steps()
        if reward_steps:
            rewards, steps = zip(*reward_steps)

        if step_idx % TEST_ITERS == TEST_ITERS - 1:
            ts = time.time()
            rewards, steps = test_net(net_act, test_env, device=device)
            print("Test done in %.2f sec, reward %.3f, steps %d" % (
                time.time() - ts, rewards, steps))
            writer.add_scalar("test_reward", rewards, step_idx)
            writer.add_scalar("test_steps", steps, step_idx)
            if best_reward is None or best_reward < rewards:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                    name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                    fname = os.path.join(save_path, name)
                    torch.save(net_act.state_dict(), fname)
                best_reward = rewards

        trajectory.append(exp)
        if len(trajectory) < TRAJECTORY_SIZE:
            continue

        traj_states = [t[0].state for t in trajectory] # remember to add the dimension of BATCH_SIZE
        traj_actions = [t[0].action for t in trajectory]
        traj_states_v = torch.FloatTensor(traj_states)
        traj_states_v = traj_states_v.to(device)
        traj_actions_v = torch.FloatTensor(traj_actions)
        traj_actions_v = traj_actions_v.to(device)
        traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device=device)
        mu_v = net_act(traj_states_v)
        old_logprob_v = calc_logprob(mu_v, net_act.logstd, traj_actions_v)

        # normalize advantages
        traj_adv_v = traj_adv_v - torch.mean(traj_adv_v)
        traj_adv_v /= torch.std(traj_adv_v)

        # drop last entry from the trajectory, an our adv and ref value calculated without it
        trajectory = trajectory[:-1]
        old_logprob_v = old_logprob_v[:-1].detach()


        sum_loss_value = 0.0
        sum_loss_policy = 0.0
        count_steps = 0

        traj_states_v = traj_states_v[:-1]
        traj_actions_v = traj_actions_v[:-1]
        # print("traj_states_v shape: ", traj_states_v.shape)
        # print("traj_ref_v shape: ", traj_ref_v.shape)
        

        for epoch in range(PPO_EPOCHES):
            for batch_ofs in range(0, len(trajectory),PPO_BATCH_SIZE):
                batch_l = batch_ofs + PPO_BATCH_SIZE
                states_v = traj_states_v[batch_ofs:batch_l]
                actions_v = traj_actions_v[batch_ofs:batch_l]
                batch_adv_v = traj_adv_v[batch_ofs:batch_l]
                batch_adv_v = batch_adv_v.unsqueeze(-1)
                batch_ref_v = traj_ref_v[batch_ofs:batch_l]
                batch_old_logprob_v = old_logprob_v[batch_ofs:batch_l]

                # critic training
                opt_crt.zero_grad()
                # print(len(states_v))
                # print(len(batch_ref_v))
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), batch_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                # actor training
                opt_act.zero_grad()
                mu_v = net_act(states_v)
                logprob_pi_v = calc_logprob(mu_v, net_act.logstd, actions_v)
                ratio_v = torch.exp(logprob_pi_v - batch_old_logprob_v)
                surr_obj_v = batch_adv_v * ratio_v
                c_ratio_v = torch.clamp(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                clipped_surr_v = batch_adv_v * c_ratio_v
                loss_policy_v = -torch.min(surr_obj_v, clipped_surr_v).mean()
                loss_policy_v.backward()
                opt_act.step()

                sum_loss_value += loss_value_v.item()
                sum_loss_policy += loss_policy_v.item()
                count_steps += 1

        trajectory.clear()
        writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
        writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
        writer.add_scalar("loss_policy", sum_loss_policy / count_steps, step_idx)
        writer.add_scalar("loss_value", sum_loss_value / count_steps, step_idx)
        print("step: %d| values: %.3f| policy_loss: %.3f| value_loss: %.3f" %\
            (step_idx, traj_ref_v.mean().item(), sum_loss_policy / count_steps, sum_loss_value / count_steps))


    # AI_agent0_net = LSTMPolicy(
    #     scope="policy0", 
    #     reuse=False, 
    #     ob_space=env.observation_space.spaces[0],
    #     ac_space=env.action_space.spaces[0],
    #     hiddens=[128,128], normalize=True
    #     )
    # params = load_from_file(param_pkl_path=path)

    # tf_config = tf.ConfigProto(
    #     inter_op_parallelism_threads=1,
    #     intra_op_parallelism_threads=1)
    # sess = tf.Session(config=tf_config)
    # sess.__enter__()
    # sess.run(tf.variables_initializer(tf.global_variables()))
    # setFromFlat(AI_agent0_net.get_variables(), params)


    # obs = env.reset()
    # max_episodes = 10
    # num_episodes = 0
    # while num_episodes < max_episodes:
    #     env.render()
    #     AI_action = AI_agent0_net.act(stochastic=True, observation=obs[0])[0]
    #     my_agent1_action = env.action_space.sample()[1]
    #     action = tuple([AI_action, my_agent1_action])
    #     next_obs, reward, done, _  = env.step(action)
    #     if done[1]:
    #         num_episodes += 1
    #         obs = env.reset()
    #     obs = next_obs