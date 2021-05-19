import gym
import gym_compete
from gym_compete.new_envs import SumoEnv
from gym_compete.new_envs.agents import *
import numpy as np
import os
import torch

library_path = "/home/chenkehan/multiagent-competition/gym-compete/gym_compete/new_envs/"

AGENT_MAP = {
        'ant': (
            os.path.join(library_path, "assets", "ant_body.xml"),
            Ant
        ),
        'humanoid': (
            os.path.join(library_path, "assets", "humanoid_body.xml"),
            Humanoid
        ),
        'humanoid_blocker': (
            os.path.join(library_path, "assets", "humanoid_body.xml"),
            HumanoidBlocker
        ),
        'humanoid_fighter': (
            os.path.join(library_path, "assets", "humanoid_body.xml"),
            HumanoidFighter
        ),
        'ant_fighter': (
            os.path.join(library_path, "assets", "ant_body.xml"),
            AntFighter
        ),
        'humanoid_kicker': (
            os.path.join(library_path, "assets", "humanoid_body.xml"),
            HumanoidKicker
        ),
        'humanoid_goalkeeper': (
            os.path.join(library_path, "assets", "humanoid_body.xml"),
            HumanoidGoalKeeper
        ),
    }

kwargs = {
    "agent_names": ["ant_fighter", "ant_fighter"],
    "world_xml_path": "/home/chenkehan/multiagent-competition/gym-compete/gym_compete/new_envs/assets/world_body_arena.xml",
    "agent_map": AGENT_MAP,
    "max_episode_steps": 200,
    "min_radius": 4.5,
    "max_radius": 5.0,
    "init_pos":[(0, 0, 2.5), (0.3, 0, 2.5)]
}

agent1_param_path = "/home/chenkehan/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl"
agent2_param_path = "/home/chenkehan/multiagent-competition/agent-zoo/sumo/ants/agent_parameters-v1.pkl"


env = SumoEnv(**kwargs)
env.reset()
action_space = env.action_space
observation_space = env.observation_space
n_agents = env.n_agents
print(action_space, '\n', observation_space)
print(isinstance(env, gym.Env))
while True:
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    # state = next_state
# env.close()
