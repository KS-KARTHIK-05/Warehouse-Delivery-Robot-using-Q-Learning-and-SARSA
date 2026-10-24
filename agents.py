"""
Tabular reinforcement-learning agents.

Each learning agent keeps a Q-table: Q[row, col, carrying, action] is how much
total future reward the agent expects if it takes `action` in that state.
The table starts at all zeros (the agent knows nothing about the warehouse)
and is corrected after every single step, using the reward it just received.
"""

import numpy as np


class TabularAgent:
    """Shared parts of both learners: the Q-table, epsilon-greedy exploration, saving."""

    name = "Tabular"

    def __init__(self, state_shape, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.996, epsilon_min=0.05, seed=None):
        self.q = np.zeros((*state_shape, n_actions))
        self.n_actions = n_actions
        self.alpha = alpha                  # learning rate: how far one update moves the estimate
        self.gamma = gamma                  # discount factor: how much future rewards count
        self.epsilon = epsilon              # exploration rate: chance of a random action
        self.epsilon_decay = epsilon_decay  # epsilon is multiplied by this after every episode
        self.epsilon_min = epsilon_min      # never stop exploring completely while training
        self.rng = np.random.default_rng(seed)

    def greedy_action(self, state):
        """Best action according to the Q-table. Ties are broken randomly."""
        q = self.q[state]
        best = np.flatnonzero(q == q.max())
        return int(best[0] if len(best) == 1 else self.rng.choice(best))

    def act(self, state):
        """Epsilon-greedy: explore with probability epsilon, otherwise exploit."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, next_action, terminated):
        raise NotImplementedError

    def save(self, path):
        np.save(path, self.q)

    def load(self, path):
        self.q = np.load(path)
        return self


class QLearningAgent(TabularAgent):
    """Q-learning: off-policy temporal-difference control (Watkins, 1989).

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    The target assumes the BEST action is taken next, whatever the agent really
    does, so it learns the value of the optimal policy even while it explores.
    """

    name = "Q-learning"

    def update(self, state, action, reward, next_state, next_action, terminated):
        future = 0.0 if terminated else self.gamma * self.q[next_state].max()
        td_error = reward + future - self.q[state][action]
        self.q[state][action] += self.alpha * td_error
        return td_error


class SarsaAgent(TabularAgent):
    """SARSA: on-policy temporal-difference control (Rummery & Niranjan, 1994).

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

    The target uses the action a' the agent will REALLY take next, random
    exploratory moves included, so it learns the value of how it actually behaves.
    """

    name = "SARSA"

    def update(self, state, action, reward, next_state, next_action, terminated):
        future = 0.0 if terminated else self.gamma * self.q[next_state][next_action]
        td_error = reward + future - self.q[state][action]
        self.q[state][action] += self.alpha * td_error
        return td_error


class RandomAgent:
    """Baseline that ignores everything and picks moves uniformly at random."""

    name = "Random"
    epsilon = 1.0

    def __init__(self, n_actions=4, seed=None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        return int(self.rng.integers(self.n_actions))

    greedy_action = act

    def update(self, *args):
        pass

    def end_episode(self):
        pass
