import numpy as np


def run_episode(env, agent, agent2):

    a1r, a2r = 0, 0 #agent 1, 2 rewards

    terminated = False
    state = env.reset()
    p = 1
    while not terminated:

        if p == 1:

            action = agent.get_action(state, env.get_moves())
            next_state, reward, terminated, info = env.step(action)
            agent.update(state, action, reward[0], terminated, next_state, env.get_moves())
            state = next_state

            if terminated: break
            p = 2

            a1r += reward[0]
                
        elif p == 2:

            action = agent2.get_action(state, env.get_moves())
            next_state, reward, terminated, info = env.step(action)
            agent2.update(state, action, reward[1], terminated, next_state, env.get_moves())
            state = next_state

            if terminated: break
            p = 1

            a2r += reward[1]
    
    a1r += reward[0]
    a2r += reward[1]
    
    return a1r, a2r

def invert_state(obs):
    """
    Invert the state of the board for the second player.
    """
    obs[obs == 1] = 3
    obs[obs == 2] = 1
    obs[obs == 3] = 2
    return obs