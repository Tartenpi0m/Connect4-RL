from BaseAgent import BaseAgent
from Net import Netv1, Netv2, LinearNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

class AgentDQNv3(BaseAgent):
    """
    Deep Q-learning agent implemented with memory replay but no target network.
    """

    def __init__(self, player, action_space, observation_space, gamma=0.99, lr=0.1, eps_init=0.5, eps_min=0.01, eps_step=1e-3, name='DeepAgent', memory_size = 500, batch_size = 64, reset_step = 1000):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.reset_step = reset_step
        super().__init__(player, action_space, observation_space, gamma, lr, eps_init, eps_min, eps_step, name)


    def reset(self, player):
        print("reset")

        self.target_step = 0

        #check if cuda is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_values = LinearNet(player).to(self.device)
        self.target_q_values = LinearNet(player).to(self.device)
        self.optimizer = optim.Adam(self.q_values.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()


        #Memory replay
        self.memory_cursor = 0
        self.memory_full = False

        self.memory_state = torch.zeros(self.memory_size, 1, 7, 6)
        self.memory_action = torch.zeros(self.memory_size)

        self.memory_state_prime = torch.zeros(self.memory_size, 1, 7, 6)
        self.memory_reward = torch.zeros(self.memory_size)
    
    def update(self, obs, action, reward, terminated, next_obs, possible_actions):
        self.optimizer.zero_grad()

        self.target_step += 1

        #Store transition (s, a, r, s') in replay memory
        if not terminated:
            self.memory_state[self.memory_cursor] = torch.tensor(obs)
            self.memory_action[self.memory_cursor] = action
            self.memory_state_prime[self.memory_cursor] = torch.tensor(next_obs)
            self.memory_reward[self.memory_cursor] = reward

            self.memory_cursor += 1
            if self.memory_cursor == self.memory_size:
                self.memory_cursor = 0
                self.memory_full = True

        #Sample a random minibatch of N transitions from the replay memory
        if self.memory_full:
            minibatch = torch.randint(0, self.memory_size, (self.batch_size,))
        else:
            minibatch = torch.randint(0, self.memory_cursor, (self.batch_size,))

        #load minibatch on cuda
        state_minibatch = self.memory_state[minibatch].to(self.device)
        action_minibatch = self.memory_action[minibatch].int().to(self.device)
        state_prime_minibatch = self.memory_state_prime[minibatch].to(self.device)
        
        #Q_value of s with action a
        current_qvalues = self.q_values.forward(state_minibatch)[torch.arange(self.batch_size), action_minibatch]
        

        with torch.no_grad():
            if not terminated:
                #Q_value of s' for max of all possible actions
                next_qvalues = self.target_q_values.forward(state_prime_minibatch)
                for i in range(7): #Set impossible actions to -1000 in order for torch.max to not selection them
                    if i not in possible_actions:
                        next_qvalues[:, i] = -1000
                target_qvalues = reward + self.gamma * torch.max(next_qvalues, dim=1)[0]
            else:
                target_qvalues = torch.ones(self.batch_size) * reward
                target_qvalues = target_qvalues.to(self.device)

        loss = self.loss(current_qvalues, target_qvalues)
        loss.backward()
        self.optimizer.step()

        if self.target_step % self.reset_step == 0:
            self.target_q_values.load_state_dict(self.q_values.state_dict())
            self.target_step = 0
            print("Target network updated")


        self.epsilon_decay()


    def eps_greedy(self, obs, possible_actions, eps=None):
        possible_actions = np.array(possible_actions)

        if eps is None:
            eps = self.eps

        if np.random.random() < eps: 
            return np.random.choice(possible_actions)
        else:
            obs = torch.tensor(obs).unsqueeze(0).to(self.device)
            b = self.q_values.forward(obs).to('cpu').detach().numpy()
            b = b.reshape(-1)
            for i in range(b.shape[0]):
                if i not in possible_actions:
                    b[i] = -np.inf
            return np.argmax(b)
            return np.random.choice(np.flatnonzero(b == np.max(b)))