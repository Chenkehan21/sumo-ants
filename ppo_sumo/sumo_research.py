import gym
import gym_compete
import numpy as np

env1 = gym.make("sumo-ants-v0")
print(env1.reset())

xml = env1.agents[0].xml
acts = xml.find('actuator')
action_dim = len(list(acts))
default = xml.find('default')
if default is not None:
    motor = default.find('motor')
    if motor is not None:
        ctrl = motor.get('ctrlrange')
        if ctrl:
            clow, chigh = list(map(float, ctrl.split()))
            high = chigh * np.ones(action_dim)
            low = clow * np.ones(action_dim)
            range_set = True
if not range_set:
    high = np.inf * np.ones(action_dim)
    low = - high
for i, motor in enumerate(list(acts)):
    ctrl = motor.get('ctrlrange')
    if ctrl:
        clow, chigh = list(map(float, ctrl.split()))
        high[i] = chigh
        low[i] = clow
print("high: ", high)
print("low: ", low)