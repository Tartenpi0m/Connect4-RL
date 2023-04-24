from typing import List
from copy import deepcopy
import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple
from colorama import Fore


class Connect4Env(gym.Env):
    """
        GameState for the Connect 4 game.
        The board is represented as a 2D array (rows and columns).
        Each entry on the array can be:
            0 = empty    (.)
            1 = player 1 (X)
            2 = player 2 (O)
        Winner can be:
             0 = No winner (yet)
             -1 = Draw
             1 = player 1 (X)
             2 = player 2 (O)
    """

    def __init__(self, width=7, height=6, connect=4, reward_step=-1, reward_win=100, reward_draw=0):
        self.num_players = 2

        self.width = width
        self.height = height
        self.connect = connect
        self.reward_step = reward_step
        self.reward_win = reward_win
        self.reward_draw = reward_draw

        # 3: Channels. Empty cells, p1 chips, p2 chips
        player_observation_space = Box(low=0, high=1,
                                       shape=(self.num_players + 1,
                                              self.width, self.height),
                                       dtype=np.int32)
        self.observation_space = Tuple([player_observation_space
                                        for _ in range(self.num_players)])
        self.action_space = Tuple([Discrete(self.width) for _ in range(self.num_players)])

        # Naive calculation. There are height * width individual cells
        # and each one can have 3 values. This is also encapsulates
        # invalid cases where a chip rests on top of an empy cell.
        self.state_space_size = 3 ** (self.height * self.width)

        self.reset()

    def reset(self) -> List[np.ndarray]:
        """
        Initialises the Connect 4 gameboard.
        """
        self.board = np.full((self.width, self.height), 0)

        self.first_player = None
        self.last_player = None
        self.winner = None

        return self.board
    
    
    def step(self, movecol, player):
        """
        Changes this GameState by "dropping" a chip in the column
        specified by param movecol.
        :param movecol: column over which a chip will be dropped
        :param player: player who is making the move (1 or 2)
        """

        if player == 1 or player == 2:
            if self.last_player is None:
                pass
            elif self.last_player == player:
                raise ValueError(f'Player {player} tried to play two times.')
        else:
            raise ValueError(f'Invalid player {player}. Should be 1 or 2.')


        if not(movecol >= 0 and movecol <= self.width and self.board[movecol][self.height - 1] == 0):
            raise IndexError(f'Invalid move. tried to place a chip on column {movecol} which is already full. Valid moves are: {self.get_moves()}')
        row = self.height - 1
        while row >= 0 and self.board[movecol][row] == 0:
            row -= 1

        row += 1

        self.board[movecol][row] = player


        self.winner, reward = self.check_for_episode_termination(movecol, row, player)

        info = {'legal_actions': self.get_moves(),
                'current_player': player}
        
        self.last_player = player

        return self.board, reward, self.winner, info
    
    def check_for_episode_termination(self, movecol, row, player):

        winner = 0
        reward = self.reward_step

        if self.does_move_win(movecol, row):
            winner = player
            if winner == 1: reward = self.reward_win
            elif winner == 2: reward = self.reward_win
        elif self.get_moves() == []:  # A draw has happened
            winner = -1
            reward = self.reward_draw
        
        return winner, reward
    
    def get_moves(self):
        """
        :returns: array with all possible moves, index of columns which aren't full
        """
        return [col for col in range(self.width) if self.board[col][self.height - 1] == 0]
    
    def does_move_win(self, x, y):
        """
        Checks whether a newly dropped chip at position param x, param y
        wins the game.
        :param x: column index
        :param y: row index
        :returns: (boolean) True if the previous move has won the game
        """
        me = self.board[x][y]
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
            p = 1
            while self.is_on_board(x+p*dx, y+p*dy) and self.board[x+p*dx][y+p*dy] == me:
                p += 1
            n = 1
            while self.is_on_board(x-n*dx, y-n*dy) and self.board[x-n*dx][y-n*dy] == me:
                n += 1

            if p + n >= (self.connect + 1): # want (p-1) + (n-1) + 1 >= 4, or more simply p + n >- 5
                return True

        return False
    
    def is_on_board(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height
    
    def render(self):
        s = ""
        for x in range(self.height - 1, -1, -1):
            for y in range(self.width):
                s += {0: Fore.WHITE + '. ', 1: Fore.RED + 'X ', 2: Fore.YELLOW + 'O '}[self.board[y][x]]
                s += Fore.RESET
            s += "\n"
        print(s)
        print()
        print(self.board)
        return self.board
