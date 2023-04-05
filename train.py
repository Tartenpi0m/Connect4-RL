import sys

#Default parameters
modelname = "default" #Folder name where tensorboard metrics will be strored for this training (inside ./runs/)
n = 1000 #Number of episodes to run
offset = 0 #Number of episodes already run from previous training


#Parameters for command line
for arg in sys.argv[1:]:
    if arg.startswith('--tensorfolder='):
        modelname = str(arg.split('=')[1])
    elif arg.startswith('--episode='):
        n = int(arg.split('=')[1])
    elif arg.startswith('--offset='):
        offset = int(arg.split('=')[1])
    elif arg.startswith('--help'):
        print("Usage: python3 train.py [--tensorfolder=foldername] [--episode=number of episodes] [--offset=number of episodes already run] [--help]")
        print("Example: python3 train.py --tensorfolder='AgentDQNv1_Netv2' --episode=10000 --offset=0")
        print("")
        print("tensorfolder: Folder name where tensorboard metrics will be strored for this training (inside ./runs/<tensorfloder>/)")
        exit()
    else:
        print(arg + " is not a valid argument. Use --help for more information")

#Import
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from envP4 import Connect4Env
from utilities import run_episode


#Select Agent
from AgentDQNv1 import AgentDQNv1 as Agent

#Tensorboard writer
writer = SummaryWriter(log_dir='./runs/' + modelname)


#Environnement initialization
env = Connect4Env(height=6, width=7, connect=4, reward_draw=0, reward_win=100, reward_step=0)


#Agent initialization
agent = Agent(1, env.action_space, env.observation_space, gamma=0.9999, lr=0.1, eps_init=0.2, eps_step=0)
agent2 = Agent(2, env.action_space, env.observation_space, gamma=0.9999, lr=0.1, eps_init=0.2, eps_step=0)





#Run training , monitor training and save model
try:
    for i in range(n):

        winner, a1r, a2r = run_episode(env, agent, agent2)

        writer.add_scalar('Agent 1 rewards', a1r, i+offset)
        writer.add_scalar('Agent 2 rewards', a2r, i+offset)
        writer.add_scalar('Agent epsilon', agent.eps, i+offset)
        #writer.add_scalar('Agent 2 epsilon', agent2.eps, i+offset)
     
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
