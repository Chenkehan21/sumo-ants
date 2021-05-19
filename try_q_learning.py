import gym
import collections


GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 100


class Agent:
    """q-learning just need q_table, we don't bother to calculate dynamic fuction.  
    """
    def __init__(self):
        self.env = gym.make("FrozenLake-v0")
        self.state = self.env.reset()
        self.q_value_table = collections.defaultdict(float)
    
    """to get s, a, r, next_s
    """
    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, done, _ = self.env.step(action)
        self.state = self.env.reset() if done else new_state
        
        return old_state, action, reward, new_state

    """we don't need to traverse all states, we just check all possible actions and choose the best one
    """
    def get_best_v_and_best_a(self, state):
        best_value, best_action = -100, 0
        for action in range(self.env.action_space.n):
            q_value = self.q_value_table[(state, action)]
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_value, best_action

    def update_q_value_table(self, s, a, r, s_next):
        old_q_value = self.q_value_table[(s, a)]
        new_q_value, _ = self.get_best_v_and_best_a(s_next)
        self.q_value_table[(s, a)] = (1 - ALPHA) * old_q_value + ALPHA * (r + GAMMA * new_q_value)

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.get_best_v_and_best_a(state)
            new_state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
            state = new_state
        
        return total_reward


def main():
    env = gym.make("FrozenLake-v0")
    iter_n = 0
    best_reward = -1
    agent = Agent()
    while True:
        iter_n += 1
        total_reward = 0.0
        s, a, r, next_s = agent.sample_env()
        agent.update_q_value_table(s, a, r, next_s)
        for i in range(TEST_EPISODES):
            total_reward += agent.play_episode(env)
        mean_reward = total_reward / TEST_EPISODES
        if mean_reward > 0.8:
            print("solved in %d iterations, final reward %.3f" % (iter_n, mean_reward))
            break
        if mean_reward > best_reward:
            print("best reward update: %.3f -> %.3f" % (best_reward, mean_reward))
            best_reward = mean_reward


if __name__ == "__main__":
    main()