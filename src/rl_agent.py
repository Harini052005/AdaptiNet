import numpy as np

class QLearningAgent:
    def __init__(self, n_states=4, n_actions=1, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.n_states = n_states   # time slots
        self.n_actions = n_actions # only 1 action (post)
        self.alpha = alpha         # learning rate
        self.gamma = gamma         # future reward importance
        self.epsilon = epsilon     # exploration
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        # epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def train(self, data, episodes=200):
        for _ in range(episodes):
            # pick random time (state)
            state = np.random.randint(0, self.n_states)

            action = self.choose_action(state)

            # reward from REAL DATA
            rewards = data[data["time"] == state]["engagement"]
            reward = rewards.mean() if len(rewards) > 0 else 0

            # next state (simulate next time)
            next_state = (state + 1) % self.n_states

            # Q-learning update
            best_next = np.max(self.Q[next_state])

            self.Q[state, action] += self.alpha * (
                reward + self.gamma * best_next - self.Q[state, action]
            )

    def best_time(self):
        return np.argmax(self.Q[:, 0])