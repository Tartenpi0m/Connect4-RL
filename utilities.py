import numpy as np
import plotly.graph_objs as go


def run_N_episodes(env, agent, agent2, N_episodes=1):



    for i in range(N_episodes):
        terminated = False
        state, _ = env.reset()
        state = obs_converter(state)
        p = 1
        while not terminated:

            if p == 1:

                action = agent.get_action(state, env.get_moves())
                next_state, reward, terminated, info = env.step(action)
                next_state = obs_converter(next_state[0])
                agent.update(state, action, reward[0], terminated, next_state)
                state = invert_state(next_state)
            
                if terminated: break
                p = 2
                    
            elif p == 2:

                action = agent2.get_action(state, env.get_moves())
                next_state, reward, terminated, info = env.step(action)
                next_state = obs_converter(next_state[1])
                agent2.update(state, action, reward[1], terminated, next_state)
                state = invert_state(next_state)

                if terminated: break
                p = 1
    return env, agent

def obs_to_string(o):
    return ''.join([str(i)[0] for i in o.flatten()])

def obs_converter(env_obs):
    """
    Convert the observation from the environment to a 2D array with 0 for empty cells, 1 for player 1 coins and 2 for player 2 coins.
    Then return the string representation of the flatten array.
    """
    obs = np.ones(env_obs[0].shape)
    for i in range(3):
        obs[env_obs[i] == 1] = i
    return obs_to_string(obs)

def invert_state(obs):
    """
    Invert the state of the board for the second player.
    """
    obs = obs.replace('1', '3')
    obs = obs.replace('2', '1')
    obs = obs.replace('3', '2')
    return obs