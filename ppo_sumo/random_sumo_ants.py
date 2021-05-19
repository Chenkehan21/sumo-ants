import gym
import gym_compete
from policy import LSTMPolicy
import tensorflow as tf
import pickle
import numpy as np


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


def get_distance(env):
    pos0 = env.agents[0].get_qpos()[:2]
    pos1 = env.agents[1].get_qpos()[:2]
    distance = np.sqrt(np.sum((pos0 - pos1)**2))
    return distance


env = gym.make("sumo-ants-v0")

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


max_episodes = 5
num_episodes = 0
total_contacts = 0
total_steps = 0
while num_episodes < max_episodes:
    print()
    agent0_total_rewards = 0.0
    agent1_total_rewards = 0.0
    n_steps = 0
    contact_steps = 0
    obs = env.reset()
    while True:
        # env.render()
        AI_action = AI_agent0_net.act(stochastic=True, observation=obs[0])[0]
        my_agent1_action = env.action_space.sample()[1]
        # AI_action = np.zeros_like(env.action_space.sample()[1])
        # my_agent1_action = np.zeros_like(env.action_space.sample()[1])
        action = tuple([AI_action, my_agent1_action])
        next_obs, reward, done, _  = env.step(action)
        distance = get_distance(env)
        print("distance: ", distance)
        if env.get_agent_contacts():
            contact_steps += 1
        # print(reward)
        n_steps += 1
        agent0_total_rewards += reward[0]
        agent1_total_rewards += reward[1]
        if done[1]:
            # print("total steps: %d" % n_steps)
            # print("contact steps %d" % contact_steps)
            avg_contacts = contact_steps / n_steps
            # print("contact_steps / n_steps: %.3f" % avg_contacts)
            # print("agent0_total_rewards: %.3f" % agent0_total_rewards)
            # print("agent1_total_rewards: %.3f" % agent1_total_rewards)
            # print()
            num_episodes += 1
            # total_contacts += avg_contacts
            total_contacts += contact_steps
            total_steps += n_steps
            break
        obs = next_obs
# env.close()
# print("agerage total steps in one episode: %.3f" % (total_steps / num_episodes))
# print("average contact steps in one episode: %.3f" % (total_contacts/num_episodes))