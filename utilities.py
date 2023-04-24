import numpy as np


def run_episode(env, agent, agent2):

    a1r, a2r = 0, 0 #agent 1, 2 rewards

    state = env.reset()
    player = np.random.choice([1, 2]) #randomly choose who goes first
    winner = 0 #False
    play_n = 0

    while not winner:
        play_n += 1
        if player == 1:
            action = agent.get_action(state, env.get_moves())
            next_state, reward, winner, info = env.step(action, player)
            a1r += reward
            agent.update(state, action, reward, winner, next_state, env.get_moves())
            state = next_state
            player = 2

        elif player == 2:
            action = agent2.get_action(state, env.get_moves())
            next_state, reward, winner, info = env.step(action, player)
            a2r += reward
            #agent2.update(state, action, reward, winner, next_state, env.get_moves())
            state = next_state
            player = 1


    if winner == 1:
        #agent2.update(state, action, -reward, winner, None, None) #agent 2 loses
        a2r += -reward
    elif winner == 2:
        agent.update(state, action, -reward, winner, None, None) #agent 1 loses
        a1r += -reward
    elif winner == -1: #if draw
        if player == 1:
            agent.update(state, action, reward, winner, None, None) 
            a1r += reward
        elif player == 2:
            #agent2.update(state, action, reward, winner, None, None) 
            a2r += reward

    return winner, a1r, a2r, play_n