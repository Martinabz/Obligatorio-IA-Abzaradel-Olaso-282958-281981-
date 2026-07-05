from math import inf
import numpy as np

from agent import Agent
from board import Board


class MinimaxAgent(Agent):
    def __init__(self, player=1, max_depth=3, heuristic="defensive"):
        super().__init__(player)
        self.max_depth = max_depth
        self.heuristic = heuristic

    def next_action(self, obs):
        action, _ = self.alpha_beta_prune(obs, self.player, self.max_depth)
        return action

    def heuristic_utility(self, board: Board):
        my_moves = len(board.get_possible_actions(self.player))
        enemy_moves = len(board.get_possible_actions(3 - self.player))
        eliminated = np.count_nonzero(board.grid == 3) #3 es como estan representadas las celdas eliminadas del tablero
        total_cells = board.board_size[0] * board.board_size[1] -2
        ratio = eliminated / total_cells


        if self.heuristic == "baseline":
            return my_moves - enemy_moves

        elif self.heuristic == "offensive":
            return my_moves - 2 * enemy_moves

        elif self.heuristic == "defensive":
            return 2 * my_moves - enemy_moves

        elif self.heuristic == "simple":
            return my_moves

        elif self.heuristic == "d2o": #defensive to offensive
            if ratio < 0.5:
                return 2 * my_moves - enemy_moves
            else:
                return my_moves - 2 * enemy_moves

        elif self.heuristic == "o2d": #offensive to defensive
            if ratio < 0.5:
                return my_moves - 2 * enemy_moves
            else:
                return 2 * my_moves - enemy_moves

        else:
            raise ValueError(f"Heurística desconocida: {self.heuristic}")

    def alpha_beta_prune(self, board: Board, player: int, depth: int):
        value, action = self.max_value(board, player, -inf, inf, depth)
        if action is None:
            actions = board.get_possible_actions(player)
            if actions:
                return actions[0], value
        return action, value

    def max_value(self, board: Board, player: int, alpha: float, beta: float, depth: int):
        terminal, winner = board.is_end(player)
        if terminal:
            return self._terminal_utility(winner), None
        if depth == 0:
            return self.heuristic_utility(board), None

        actions = board.get_possible_actions(player)
        if not actions:
            return self._terminal_utility(3 - player), None

        value = -inf
        best_action = None
        for action in actions:
            next_board = board.clone()
            next_board.play(action, player)
            next_value, _ = self.min_value(next_board, 3 - player, alpha, beta, depth - 1)
            if best_action is None or next_value > value:
                value = next_value
                best_action = action
            if value >= beta:
                return value, best_action
            alpha = max(alpha, value)

        return value, best_action

    def min_value(self, board: Board, player: int, alpha: float, beta: float, depth: int):
        terminal, winner = board.is_end(player)
        if terminal:
            return self._terminal_utility(winner), None
        if depth == 0:
            return self.heuristic_utility(board), None

        actions = board.get_possible_actions(player)
        if not actions:
            return self._terminal_utility(3 - player), None

        value = inf
        best_action = None
        for action in actions:
            next_board = board.clone()
            next_board.play(action, player)
            next_value, _ = self.max_value(next_board, 3 - player, alpha, beta, depth - 1)
            if best_action is None or next_value < value:
                value = next_value
                best_action = action
            if value <= alpha:
                return value, best_action
            beta = min(beta, value)

        return value, best_action

    def _terminal_utility(self, winner: int):
        if winner == self.player:
            return inf
        return -inf
