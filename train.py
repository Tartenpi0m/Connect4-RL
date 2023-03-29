import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import sys
from Agent import CNNAgent, BaseAgent
from env_p4 import Connect4Env
from utilities import run_episode

import tensorboard_logger as tflog



tensorfolder = None

if len(sys.argv) > 3:
    modelname = sys.argv[1] # Name of the tensorboard folder
    n = int(sys.argv[2]) # Number of episodes to run
    offset = int(sys.argv[3]) #Number of episodes already run from previous training
else:
    raise ValueError("Please provide a tensorboard folder name and number of episodes to run\n \
                     Example: python3 train.py 'Qlearning' 10000 0")

writer = SummaryWriter(log_dir='./runs/' + modelname)


env = Connect4Env(height=6, width=7, connect=4, reward_draw=0, reward_win=100, reward_step=0, )
lr = 0.1
gamma = 0.9999



agent = CNNAgent(env.action_space, env.observation_space, gamma=gamma, lr=lr, eps_init=0.2, eps_step=1e-5)
agent2 = CNNAgent(env.action_space, env.observation_space, gamma=gamma, lr=lr, eps_init=1, eps_step=0)

try:
    for i in range(n):

        a1r, a2r = run_episode(env, agent, agent2)

        writer.add_scalar('Agent 1 rewards', a1r, i+offset)
        writer.add_scalar('Agent 2 rewards', a2r, i+offset)
        writer.add_scalar('Agent 1 epsilon', agent.eps, i+offset)
        writer.add_scalar('Agent 2 epsilon', agent2.eps, i+offset)
     
        if i % 100 == 0:
            print(i)
            writer.flush()

        if i % 1000 == 0:
            print("Saving model")
            torch.save(agent.q_values, './runs/' + modelname  + '/' + str(i) + '_a1' + '.pt')
            torch.save(agent2.q_values, './runs/' + modelname  + '/' + str(i) + '_a2' + '.pt')
    
    torch.save(agent.q_values, './runs/' + modelname  + '/' + str(n) + '_a1' + '.pt')
    torch.save(agent2.q_values, './runs/' + modelname  + '/' + str(n) + '_a2' + '.pt')

except KeyboardInterrupt:
    
    torch.save(agent.q_values, './runs/' + modelname  + '/' + str(i) + '_a1' + '.pt')
    torch.save(agent2.q_values, './runs/' + modelname  + '/' + str(i) + '_a2' + '.pt')
