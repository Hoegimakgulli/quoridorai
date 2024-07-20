import numpy as np
import random
from time import time
from temp_env import Quoridor


class MCTSNode:
    def __init__(self, state: Quoridor, parent=None, move=None):
        self.state: Quoridor = state
        self.parent: MCTSNode = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, c_param=1.4):
        choices_weights = [(c.wins / c.visits) + c_param * np.sqrt((2 * np.log(self.visits) / c.visits)) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        legal_moves = self.state.get_legal_moves()
        tried_moves = [child.move for child in self.children]
        move = random.choice([move for move in legal_moves if move not in tried_moves])

        child_node = MCTSNode(self.state.make_move(move), self, move)
        self.children.append(child_node)
        return child_node

    def rollout(self):
        current_rollout_state = self.state
        while current_rollout_state.check_winner() == 0:
            possible_moves = current_rollout_state.get_legal_moves()
            action = random.choice(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.get_reward()

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


class MCTS:
    def __init__(self, state, n_playout=1000, time_limit=None):
        self.root = MCTSNode(state)
        self.n_playout = n_playout
        self.time_limit = time_limit
        self.time = 0
        self.now = time.now()

    def _playout(self):
        node = self.root
        while not node.state.is_terminal():
            if node.is_fully_expanded():
                node = node.best_child()
            else:
                node = node.expand()
                break
        result = node.rollout()
        node.backpropagate(result)

    def get_move(self):
        for _ in range(self.n_playout):
            self._playout()
            if self.time_limit != None:
                self.time += time.now() - self.now
                self.now = time.now()
                if self.time >= self.time_limit:
                    return self.root.best_child.move
        return self.root.best_child
