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
            return np.random.choice(np.flatnonzero(b == np.max(b)))
        
    def get_action(self, obs, possible_actions, eps=None): 
        return self.eps_greedy(obs, possible_actions, eps)
        
    def update(self, obs, action, reward, terminated, next_obs):
        
        raise NotImplementedError
        
    def epsilon_decay(self):
        self.eps = max(self.eps - self.eps_step, self.eps_min)
        
    def reset(self):
        raise NotImplementedError
        
class CNNAgent(BaseAgent):


    def __init__(self, player, action_space, observation_space, gamma=0.99, lr=0.1, eps_init=0.5, eps_min=0.01, eps_step=1e-3, name='DeepAgent'):
        super().__init__(player, action_space, observation_space, gamma, lr, eps_init, eps_min, eps_step, name)


    def reset(self, player):

        self.q_values = Net(player)
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


class Net(nn.Module):
    def __init__(self, player):
        super(Net, self).__init__()
        if player == 2:
            self.adapt = self.invert_state
        else:
            self.adapt = self.nothin
        self.pad1 = nn.ZeroPad2d((2,2,2,2))
        self.pad1.padding_mode = 'constant'
        self.pad1.value = -5

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 5, stride = 1)
        self.fc = nn.Linear(8*6*7, 7)
    
    def forward(self, x):
        x = self.adapt(x.copy())
        x = torch.tensor(x).reshape(1, 6, 7).float()
        x[x == 2] = -1 #replace 2 (player 2 coins) with -1
        x = self.pad1(x)
        x = F.relu(self.conv1(x))
        x = x.view(-1, 8*6*7)
        x = self.fc(x)
        return x.view(-1)
    
    def invert_state(self, obs):
        """
        Invert the state of the board for the second player.
        """
        obs[obs == 1] = 3
        obs[obs == 2] = 1
        obs[obs == 3] = 2
        return obs
    
    def nothin(self, obs):
        return obs
   
