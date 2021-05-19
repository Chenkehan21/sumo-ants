import ptan
import numpy as np
import torch
import math
import tensorflow as tf
import pickle
from policy import LSTMPolicy


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


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs0, obs1 = env.reset()

        AI_agent0_net = LSTMPolicy(
                scope="policy0", 
                reuse=True, 
                ob_space=env.observation_space.spaces[0],
                ac_space=env.action_space.spaces[0],
                hiddens=[128,128], normalize=True
        )
        params = load_from_file(param_pkl_path=path)

        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        sess = tf.Session(config=tf_config)
        sess.__enter__()
        sess.run(tf.variables_initializer(tf.global_variables()))
        setFromFlat(AI_agent0_net.get_variables(), params)


        while True:
            obs1_v = ptan.agent.float32_preprocessor([obs1]).to(device)
            mu_v = net(obs1_v)[0]
            agent1_action = mu_v.squeeze(dim=0).data.cpu().numpy()
            agent1_action = np.clip(agent1_action, -1, 1)
            agent0_action = AI_agent0_net.act(stochastic=True, observation=obs0)[0]
            if np.isscalar(agent1_action): 
                agent1_action = [agent1_action]
            obs, reward, done, _ = env.step(tuple([agent0_action, agent1_action]))
            obs0, obs1 = obs
            rewards += reward[1]
            steps += 1
            if done:
                break
    return rewards / count, steps / count

def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2



