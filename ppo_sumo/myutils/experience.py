import gym
import torch
import random
import collections
from torch.autograd import Variable

import numpy as np

from collections import namedtuple, deque

from ptan.agent import BaseAgent
from ptan.common import utils
from policy import LSTMPolicy
import tensorflow as tf
import pickle


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


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


class MAExperienceSource:
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        states0, states1, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], [], []
        env_lens = []
        path = "/home/chenkehan/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl"
        for env in self.pool:
            # in multiagent environment obs is a tuple with length equals to number of agents
            obs = env.reset() 
            AI_agent0_net = LSTMPolicy(
                scope="policy0", 
                reuse=False, 
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
            
            if self.vectorized:
                obs_len = len(obs)
                states1.extend(obs)
            else:
                obs_len = 1
                states1.append(obs[1])
                states0.append(obs[0])
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_state())

        iter_idx = 0
        while True:
            agent1_actions = [None] * len(states1)
            agent0_actions = [None] * len(states1)

            states_input = []
            states_indices = []
            for idx, (state0, state1) in enumerate(zip(states0, states1)):
                agent0_actions[idx] = AI_agent0_net.act(stochastic=True, observation=state0)[0]
                if state1 is None:
                    agent1_actions[idx] = self.pool[0].action_space.sample()[1]
                else:
                    states_input.append(state1)
                    states_indices.append(idx)
            if states_input:
                states_actions, new_agent_states = self.agent(states_input, agent_states)
                for idx, action in enumerate(states_actions):
                    g_idx = states_indices[idx]
                    agent1_actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            grouped_agent1_actions = _group_list(agent1_actions, env_lens)
            grouped_agent0_actions = _group_list(agent0_actions, env_lens)
            # print("gourped_agent0_actions: ", grouped_agent0_actions)
            # print("agent0_actions: ", agent0_actions)
            # print("agent1_actions: ", agent1_actions)

            global_ofs = 0
            for env_idx, (env, action_agent1, action_agent0) in enumerate(zip(self.pool, grouped_agent1_actions, grouped_agent0_actions)):
                if self.vectorized:
                    next_state_n, r_n, is_done_n, _ = env.step(tuple([action_agent0, action_agent1]))
                else:
                    # print("action_agent0: ", action_agent0)
                    # print("action_agent1: ", action_agent1)
                    next_state, r, is_done, _ = env.step(tuple([action_agent0[0], action_agent1[0]]))
                    next_state_n, r_n, is_done_n = [next_state[1]], [r[1]], [is_done[1]] # in order to use enumerate to get index later!
                    next_state_n0 = [next_state[0]]

                for ofs, (action, next_state, r, is_done, next_state0) in enumerate(zip(action_agent1, next_state_n, r_n, is_done_n, next_state_n0)):
                    idx = global_ofs + ofs
                    state1 = states1[idx]
                    history = histories[idx]

                    cur_rewards[idx] += r
                    cur_steps[idx] += 1
                    if state1 is not None:
                        history.append(Experience(state=state1, action=action, reward=r, done=is_done))
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    states1[idx] = next_state
                    states0[idx] = next_state0
                    if is_done:
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.steps_count:
                            yield tuple(history)
                        # generate tail of history
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        self.total_rewards.append(cur_rewards[idx])
                        self.total_steps.append(cur_steps[idx])
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        if not self.vectorized:
                            states1[idx], states0[idx] = env.reset()
                        else:
                            states0[idx], states1[idx] = None, None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                global_ofs += len(action_agent1)
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res

def _group_list(items, lens):
    res = []
    cur_ofs = 0
    for g_len in lens:
        res.append(items[cur_ofs:cur_ofs+g_len])
        cur_ofs += g_len
    return res