from BaseAgent import BaseAgent

import numpy as np

class CounterAgent(BaseAgent):
    """
    First Deep Q-learning agent implemented with no memory replay and no target network.
    """

    def __init__(self, player, action_space, observation_space, gamma=0.99, lr=0.1, eps_init=0.5, eps_min=0.01, eps_step=1e-3, name='CounterAgent'):
        super().__init__(player, action_space, observation_space, gamma, lr, eps_init, eps_min, eps_step, name)
        self.p = player

    def reset(self,palyer):
        self.prev_board_state = [[0 for j in range(6)] for i in range(7)]
    

    def eps_greedy(self, obs, possible_actions, eps=None):
        self.board_state = np.array(obs)
        # Find the column where the opponent played
        diff = self.board_state - self.prev_board_state

        nonzero_idx = np.where(diff != 0) 

        if len(nonzero_idx[1]) > 0:

            opponent_last_move = nonzero_idx[0][0]

        else:
            # If the difference is all zeros, assume opponent didn't make a move
            opponent_last_move = -1

        for row in range(0,self.board_state.shape[1]-1, 1):
            if opponent_last_move >= 0 and self.board_state[opponent_last_move][row] == 0:
                movecol = opponent_last_move
                break
            
            else :
                movecol =  np.random.choice(possible_actions)

        self.prev_board_state = self.new_board(movecol,self.p)

        return movecol

    def new_board(self,movecol,player):
        if not(movecol >= 0 and movecol <= 7 and self.board_state[movecol][6 - 1] == 0):
            raise IndexError(f'Invalid move. tried to place a chip on column {movecol} which is already full. Valid moves are: {self.get_moves()}')
        row = 6 - 1

        while row >= 0 and self.board_state[movecol][row] == 0:
            row -= 1

        row += 1

        self.board_state[movecol][row] = player

        return self.board_state