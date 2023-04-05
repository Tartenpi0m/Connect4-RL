import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Netv2(nn.Module):
    """
    Second Network tested with 3 convolutional layer and 2 fully connected layer.
    """
    def __init__(self, player):
        super().__init__()

        #adapt the state of the board for the second player (the environnement is not symmetric the network is trained for player 1)
        if player == 2:
            self.adapt = self.invert_state
        else:
            self.adapt = self.nothin

        #Custom padding to avoid the border effect
        self.pad1 = nn.ZeroPad2d((2,2,2,2))
        self.pad1.padding_mode = 'constant'
        self.pad1.value = -5

        #define the network
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 5, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 5, stride = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1)
        self.fc1 = nn.Linear(32*6*7, 512)
        self.fc2 = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.adapt(x.copy())
        x = torch.tensor(x).reshape(1, 6, 7).float()
        x[x == 2] = -1 #adapt player 2 chip to make sens for the network (2 replace by -1)
        x = self.pad1(x)
        x = F.relu(self.conv1(x))
        x = self.pad1(x)
        x = F.relu(self.conv2(x))
        x = self.pad1(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32*6*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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

class Netv1(nn.Module):
    """
    First Network tested with 1 convolutional layer and 1 fully connected layer only.
    """
    def __init__(self, player):
        super(Netv1, self).__init__()

        #adapt the state of the board for the second player (the environnement is not symmetric the network is trained for player 1)
        if player == 2:
            self.adapt = self.invert_state
        else:
            self.adapt = self.nothin

        #Custom padding to avoid the border effect
        self.pad1 = nn.ZeroPad2d((2,2,2,2))
        self.pad1.padding_mode = 'constant'
        self.pad1.value = -5

        #define the network
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 5, stride = 1)
        self.fc = nn.Linear(8*6*7, 7)
    
    def forward(self, x):
        x = self.adapt(x.copy())
        x = torch.tensor(x).reshape(1, 6, 7).float()
        x[x == 2] = -1 #adapt player 2 chip to make sens for the network (2 replace by -1)
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