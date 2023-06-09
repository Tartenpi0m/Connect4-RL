from BaseAgent import BaseAgent
from Net import Netv1, Netv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AgentDQNv1(BaseAgent):
    """
    First Deep Q-learning agent implemented with no memory replay and no target network.
    """

    def __init__(self, player, action_space, observation_space, gamma=0.99, lr=0.1, eps_init=0.5, eps_min=0.01, eps_step=1e-3, name='DeepAgent'):
        super().__init__(player, action_space, observation_space, gamma, lr, eps_init, eps_min, eps_step, name)


    def reset(self, player):

        self.q_values = Netv2(player)
        self.optimizer = optim.Adam(self.q_values.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
    
    def update(self, obs, action, reward, terminated, next_obs, possible_actions):
        self.optimizer.zero_grad()

        if not terminated:
            q_prime = self.q_values.forward(next_obs) #Q(s', pour tout a)
            a_prime = self.get_action(next_obs,  possible_actions, eps=0) #a'
            y = reward + self.gamma * q_prime[a_prime] #y = r + gamma * Q(s', a')

            y_hat = self.q_values.forward(obs)[action] #y_hat = Q(s, a)
            
            loss = self.loss(y_hat, y)
        else:
            y = torch.tensor(reward).float()
            y_hat = self.q_values.forward(obs)[action]
            loss = self.loss(y_hat, y)
        
        loss.backward()
        self.optimizer.step()

        self.epsilon_decay()