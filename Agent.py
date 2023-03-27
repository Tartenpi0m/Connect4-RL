import numpy as np
from collections import defaultdict

class QLearner(): 
    """
        Stores the data and computes the observed returns.
    """
    q_values = None

    def __init__(self, 
                 player,# 1 or 2
                 action_space, 
                 observation_space, 
                 gamma=0.99, 
                 lr=0.1,
                 eps_init=.5, 
                 eps_min=1e-5,
                 eps_step=1-3,
                 name='Q-learning'):
        
        self.player = player
        self.player_index = player - 1
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.lr = lr
        
        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_step = eps_step

        self.name = name
        
        self.reset()
    
    def eps_greedy(self, obs, possible_actions, eps=None):
        possible_actions = np.array(possible_actions)

        if eps is None: 
            eps = self.eps

        if np.random.random() < self.eps: 
            return np.random.choice(possible_actions)
            #return self.action_space[self.player_index].sample()
        else:
            b = QLearner.q_values[obs]
            b = b[np.isin(np.arange(len(b)), possible_actions)]
            return np.random.choice(np.flatnonzero(b == np.max(b))) # argmax with random tie-breaking
        
    def get_action(self, obs, possible_actions): 
        return self.eps_greedy(obs, possible_actions)
        
    def update(self, obs, action, reward, terminated, next_obs):
        if not terminated:
            QLearner.q_values[obs][action] += self.lr * (reward + self.gamma * np.max(QLearner.q_values[next_obs]) - QLearner.q_values[obs][action])
        else: #if final state, there is no next state
            QLearner.q_values[obs][action] += self.lr * (reward - QLearner.q_values[obs][action])

        self.epsilon_decay()
        
    def epsilon_decay(self):
        self.eps = max(self.eps - self.eps_step, self.eps_min)
        
    def reset(self):
        QLearner.q_values = defaultdict(lambda: np.zeros(self.action_space[self.player_index].n))