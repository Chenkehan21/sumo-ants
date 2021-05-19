import gym
import collections
from tensorboardX import SummaryWriter


"""
Reward table: A dictionary with the composite key "source state" + "action": r(s, a). 
The value is obtained from the immediate reward.

Transitions table: A dictionary keeping counters of the experienced transitions. 
The key is the composite "state" + "action", and the value is another dictionary 
that maps the target state into a count of times that we have seen it. For example, 
if in state 0 we execute action 1 ten times, after three times it will lead us 
to state 4 and after seven times to state 5.The entry with the key (0, 1) in this 
table will be a dict with contents {4: 3, 5: 7}. We can use this table to estimate
the probabilities of our transitions.

Value table: A dictionary that maps a state into the calculated value of this state.
"""


RANDOM_STEPS = 500
TEST_EPISODES = 100
DISCOUNT = 0.9


class Agent:
    def __init__(self):
        self.env = gym.make("FrozenLake-v0")
        self.state = self.env.reset()
        self.reward_table = collections.defaultdict(float)
        self.transition_table = collections.defaultdict(collections.Counter)
        self.value_table = collections.defaultdict(float)

    def play_n_random_episodes(self, n=RANDOM_STEPS):
        for i in range(n):
            action = self.env.action_space.sample()
            new_state, reward, done, _ = self.env.step(action)
            self.reward_table[(self.state, action)] = reward
            self.transition_table[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if done else new_state

    """when calculate action value q(s, a) state and action are fixed. 
    we only need to summarize in field target states and reward
    we need to know the dynamic function and we can get it indirectly from transition table
    """
    def calc_q_value(self, state, action):
        target_counter = self.transition_table[(state, action)] # it's a dictionary!
        total = sum(target_counter.values())
        q_value = 0.0
        for target_state, count in target_counter.items():
            probability = count / total
            reward = self.reward_table[(state, action)]
            val = reward + DISCOUNT * self.value_table[target_state]
            q_value += probability * val
        return q_value

    """value iteration should traverse all states.
    """
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            q_value_table = [self.calc_q_value(state, action) for action in range(self.env.action_space.n)]
            self.value_table[state] = max(q_value_table)

    """greedy policy choose action for a state.
    """
    def select_action(self, state):
        best_value, best_action = -100, 0
        for action in range(self.env.action_space.n):
            q_value = self.calc_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action

    """This function is used to play test episodes, during which we don't want to 
    mess with the current state of the main environment used to gather random data. 
    So, we use the second environment passed as an argument.
    """
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
            else:
                state = next_state
        return total_reward


def main():
    env = gym.make("FrozenLake-v0")
    agent = Agent()
    iter_n = 0
    best_reward = -100
    while True:
        iter_n += 1
        total_reward = 0.0
        agent.play_n_random_episodes()
        agent.value_iteration()
        for i in range(TEST_EPISODES):
            total_reward += agent.play_episode(env)
        mean_reward = total_reward / TEST_EPISODES
        if mean_reward >= 0.8:
            print("solved in iteration %d, reward: %.3f" % (iter_n, mean_reward))
            break
        if mean_reward > best_reward:
            print("best reward update: %.3f -> %.3f" % (best_reward, mean_reward))
            best_reward = mean_reward


if __name__ == "__main__":
    main()