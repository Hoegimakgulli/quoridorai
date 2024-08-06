import numpy as np
import random
import time
from temp_env import Quoridor
from ai_test import DQNAgent
import torch


class MCTSNode:
    def __init__(self, state: Quoridor, brain: DQNAgent, epsilon, parent=None, move=None):
        self.state: Quoridor = state.clone()
        self.brain: DQNAgent = brain
        self.epsilon = epsilon
        self.parent: MCTSNode = parent
        self.move = move
        self.team = 1 if parent is None or self.parent.team == 2 else 2
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.wins = 0

        self.evaluated_moves = self.brain.get_action(self.state.get_board(), self.state.get_legal_moves(1, 0), self.epsilon)

    def is_fully_expanded(self):
        if self.team == 1:
            return len(self.children) == len(self.state.get_legal_moves(1, 0))
        else:
            return len(self.children) == 1

    def best_child(self, c_param=1.4):
        choices_weights = [(c.wins / c.visits) + c_param * np.sqrt((2 * np.log(self.visits) / c.visits)) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        if self.team == 1:
            max_move = self.state.get_legal_moves(1, 0)[self.evaluated_moves.index(sorted(self.evaluated_moves, reverse=True)[len(self.children)])]
            index = 0
        else:
            max_move, index = self.state.enemy_turn(do_move=False)
        self.state.move(index, max_move, self.team)
        child_node = MCTSNode(self.state, self.brain, self.epsilon, self, max_move)
        self.children.append(child_node)
        return child_node

    def rollout(self):
        current_rollout_state = self.state
        while current_rollout_state.check_winner() == 0:
            possible_moves = current_rollout_state.get_legal_moves(1, 0)
            action = random.choice(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.get_reward()

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


class MCTS:
    def __init__(self, state, brain: DQNAgent, epsilon, n_playout=1000, time_limit=None):
        self.root: MCTSNode = MCTSNode(state, brain, epsilon)
        self.n_playout = n_playout
        self.time_limit = time_limit
        self.time = 0
        self.now = time.time()

    def _playout(self):
        node = self.root
        while node.state.check_winner() == 0:
            if node.is_fully_expanded():
                node = node.best_child()
            else:
                node = node.expand()
                break
        # result = node.rollout()
        result = node.state.get_reward()
        node.backpropagate(result)

    def get_move(self):
        for _ in range(self.n_playout):
            self._playout()
            if self.time_limit != None:
                self.time += time.time() - self.now
                self.now = time.time()
                if self.time >= self.time_limit:
                    return self.root.best_child.move
        return self.root.best_child.move
