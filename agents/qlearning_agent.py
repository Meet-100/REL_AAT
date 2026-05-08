import numpy as np
import random
import pickle

class QLearningAgent:
    """
    Q-Learning Agent for water distribution optimization.
    Uses a Q-table to store state-action values.
    """
    def __init__(self, n_actions, learning_rate=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dictionary mapping state -> list of action values
        self.q_table = {}

    def get_q_values(self, state):
        """Return Q-values for a state, initializing if necessary."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state):
        """Choose action using epsilon-greedy strategy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.n_actions - 1) # Explore
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values) # Exploit

    def update(self, state, action, reward, next_state):
        """Update Q-table using the Bellman equation."""
        current_q = self.get_q_values(state)[action]
        max_next_q = np.max(self.get_q_values(next_state))
        
        # Q-learning formula: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, file_path):
        """Save Q-table to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, file_path):
        """Load Q-table from a file."""
        with open(file_path, 'rb') as f:
            self.q_table = pickle.load(f)
            self.epsilon = self.epsilon_min # Set to min for evaluation
