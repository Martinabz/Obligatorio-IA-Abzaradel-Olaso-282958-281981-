from math import inf

from agent import Agent
from board import Board


class ExpectimaxAgent(Agent):
    def __init__(self, player=1, max_depth=3):
        super().__init__(player)
        self.max_depth = max_depth

    def next_action(self, obs):
        action, _ = self.expectimax(obs, self.player, self.max_depth)
        if action is None:
            actions = obs.get_possible_actions(self.player)
            return actions[0] if actions else None
        return action

    def heuristic_utility(self, board: Board):
        my_moves = len(board.get_possible_actions(self.player))
        enemy_moves = len(board.get_possible_actions(3 - self.player))
        return my_moves - enemy_moves

    def expectimax(self, board: Board, player: int, depth: int):
        value, action = self.max_value(board, player, depth)
        return action, value

    def max_value(self, board: Board, player: int, depth: int):
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
            next_value, _ = self.exp_value(next_board, 3 - player, depth - 1)
            if best_action is None or next_value > value:
                value = next_value
                best_action = action

        return value, best_action

    def exp_value(self, board: Board, player: int, depth: int):
        terminal, winner = board.is_end(player)
        if terminal:
            return self._terminal_utility(winner), None
        if depth == 0:
            return self.heuristic_utility(board), None

        actions = board.get_possible_actions(player)
        if not actions:
            return self._terminal_utility(3 - player), None

        total = 0.0
        count = 0
        for action in actions:
            next_board = board.clone()
            next_value, _ = self.max_value(next_board, self.player, depth - 1)
            total += next_value
            count += 1

        if count == 0:
            return self._terminal_utility(3 - player), None

        return total / count, None

    def _terminal_utility(self, winner: int):
        if winner == self.player:
            return inf
        return -inf
