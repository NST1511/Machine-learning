from typing import Optional, Union, Tuple, List

import numpy as np


class TicTacToe:
    def __init__(self):
        # value mappings
        self._mappings = {
            -1: None,
            0: 'x',
            1: 'o',
            'x': 0,
            'o': 1
        }

        self.reset_game_state()

    def play_x(self, row: int, col: int):
        self._make_move((row, col), self._mappings['x'])

    def play_o(self, row: int, col: int):
        self._make_move((row, col), self._mappings['o'])

    def random_play_x(self) -> (int, int):
        return self._random_play('x')

    def random_play_o(self) -> (int, int):
        return self._random_play('o')

    def _random_play(self, player: str) -> (int, int):
        # extract player
        ext_player = self._mappings[player]

        possible_moves = self.get_possible_moves()
        random_move = np.random.randint(0, len(possible_moves))
        self._make_move(possible_moves[random_move], ext_player)
        return possible_moves[random_move]

    def get_possible_moves(self):
        """
        Returns the possible moves to make at the current state of the game.
        :return: a list of tuples (row, column) indicating the possible positions to choose.
        """
        possible_moves = []
        for row_num, row_val in enumerate(self._board):
            for cell_num, cell_val in enumerate(row_val):
                if cell_val == -1:
                    possible_moves.append((row_num, cell_num))
        return possible_moves

    def _make_move(self, cell: (int, int), player: int):
        curr_val = self._board[cell[0], cell[1]]
        if curr_val != -1:
            raise ValueError('Cannot make move on non-empty cell!')

        if self._last_move == player:
            raise ValueError('One player cannot make two moves in a row!')

        self._board[cell[0]][cell[1]] = player
        self._last_move = player

    def check_for_winner(self) -> Union[None, str]:
        win_x = self._determine_winner(self._mappings['x'])
        win_o = self._determine_winner(self._mappings['o'])

        if win_x is not None:
            return self._mappings[win_x]
        elif win_o is not None:
            return self._mappings[win_o]
        else:
            return None

    def check_for_board_filled(self) -> bool:
        return np.all(self._board != -1)

    def _determine_winner(self, player: int) -> Optional[int]:
        win_vec = np.array([player, player, player], dtype=np.int)

        # check rows
        for i in range(3):
            if np.all(self._board[i] == win_vec):
                return player

        # check cols
        for i in range(3):
            if np.all(self._board[:, i] == win_vec):
                return player

        # check diagonals
        if np.all(np.diag(self._board) == win_vec) or np.all(np.diag(np.rot90(self._board)) == win_vec):
            return player

        return None

    def get_board(self):
        return self._board

    def get_last_move(self):
        return self._last_move

    def print_board(self):
        # create string output
        return_str = ['_____________']
        for row in self._board:
            return_str.append('| {} | {} | {} |'.format(row[0], row[1], row[2]))
        return_str.append('_____________')

        final_str = '\n'.join(return_str)
        final_str = final_str.replace('-1', ' ')
        final_str = final_str.replace('0', self._mappings[0])
        final_str = final_str.replace('1', self._mappings[1])
        print(final_str)

    def reset_game_state(self):
        self._board = np.tile(-1, 9).reshape((3, 3))
        self._last_move = None


def transform_move(row: int, col: int) -> np.ndarray:
    play = np.ones((9,)) * -1
    play[row * 3 + col] = 0
    return play


reward_scaling_factor = 100


def get_rewards(winner: str, collected_plays: List[Tuple[Tuple[int, int], np.ndarray]]) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Computes the rewards in combination with the resulting training data for one match.
    :param winner: string indicating the winner of the game. Either 'x', 'o' or None for a draw.
    :param collected_plays: a list of all plays of player x where each list element is a tuple (chosen_cell, board_state_before_play)
    :return: a tuple with rewards (target data) and board state/chosen cell (training data)
    """
    if winner == 'x':
        final_reward = 1 * reward_scaling_factor
    elif winner == 'o':
        final_reward = -1 * reward_scaling_factor
    else:
        # draw
        final_reward = 0

    # discount rewards
    dis_rewards = []
    dis_boards = []
    dis_plays = []
    discount_factor = 0.8
    for i, state in enumerate(collected_plays):
        dis_rewards.append(discount_factor ** ((len(collected_plays) - 1) - i) * final_reward)
        dis_boards.append(state[1].flatten())
        dis_plays.append(transform_move(state[0][0], state[0][1]))

    return np.array(dis_rewards), np.concatenate([np.stack(dis_boards), np.stack(dis_plays)], axis=1)
