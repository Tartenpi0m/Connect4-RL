# Connect4-RL

## Training

Execute the python file train.py to train an agent.   
`python3 train.py --help` to get more info.
To select your agent you need to modify the import directly in the code of train.py

## Create Agent

Just create a class which inherite from Base Agent.  

## Monitor the training

Execute tensorboard to monitor metrics (reward, epsilon, and number of plays) while training.  
`tensorboard --logdir="/runs/"`
