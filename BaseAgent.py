import numpy as np
from collections import defaultdict

#CNN model for Q-learning for Connect4 with Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BaseAgent(): 
    """
        Stores the data and computes the observed returns.
    """

    def __init__(self, 
                 player, 
                 action_space, 
                 observation_space, 
                 gamma=0.99, 
                 lr=0.1,
                 eps_init=.5, 
                 eps_min=1e-5,
                 eps_step=1e-3,
                 name='AbstractBaseAgent'):
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.lr = lr
        
        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_step = eps_step

        self.name = name
        self.q_values = None
        
        self.reset(player)
    
    def eps_greedy(self, obs, possible_actions, eps=None):
        possible_actions = np.array(possible_actions)

        if eps is None:
            eps = self.eps

        if np.random.random() < eps: 
            return np.random.choice(possible_actions)
        else:
            b = self.q_values.forward(obs).detach().numpy()
            for i in range(len(b)):
                if i not in possible_actions:
                    b[i] = -np.inf
            return np.argmax(b)
            return np.random.choice(np.flatnonzero(b == np.max(b)))
        
    def get_action(self, obs, possible_actions, eps=None): 
        return self.eps_greedy(obs, possible_actions, eps)
        
    def update(self, obs, action, reward, terminated, next_obs):
        
        raise NotImplementedError
        
    def epsilon_decay(self):
        self.eps = max(self.eps - self.eps_step, self.eps_min)
        
    def reset(self):
        raise NotImplementedError



   
