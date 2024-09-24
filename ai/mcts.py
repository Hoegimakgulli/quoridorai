import numpy as np
import random
from utils import Vector2
from environment import Quoridor
from model import DQNAgent
import math

##TODO: AGENT 개발 후 추가


class MCTSNode:
    def __init__(self, state: Quoridor, move: Vector2, parent=None):
        self.state: Quoridor = state
        self.move: Vector2 = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.available_moves())

    def best_child(self, exploration_weight=1.4):
        choices_weights = [(child.value / child.visits) + exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits) for child in self.children]
        return self.children[choices_weights.index(max(choices_weights))]


class MCTS:
    def __init__(self, game: Quoridor, brain: DQNAgent, simulations=100):
        self.game: Quoridor = game
        self.brain: DQNAgent = brain
        self.simulations = simulations

    def search(self, initial_state):  # 탐색
        root = MCTSNode(initial_state, Vector2(4, 0))

        for _ in range(self.simulations):
            node = self.select(root)
            reward = self.simulate(node.state)
            self.backpropagate(node, reward)

        return root.best_child(exploration_weight=0)

    def select(self, node: MCTSNode):  # 선택: 가장 유망한 노드 선택
        while node.state.check_winner() == 0 and node.is_fully_expanded():  # 노드가 완전 확장되지 않았거나, 승자가 나오지 않았을 때
            node = node.best_child()  # 가장 유망한 노드 선택
        if not node.is_fully_expanded():  # 노드가 완전 확장되지 않았을 때
            return self.expand(node)  # 확장
        return node

    def expand(self, node: MCTSNode):  # 확장: 노드 확장
        moves = node.state.get_movable_positions()  # 이동 가능한 위치
        for move in moves:  # 이동 가능한 위치에 대해
            new_state = node.state.clone()  # 상태 복사
            new_state.auto_turn(move_position=move)  # 턴 진행
            child_node = MCTSNode(new_state, move, parent=node)  # 자식 노드 생성
            node.children.append(child_node)  # 자식 노드 추가
        return random.choice(node.children)  # 랜덤으로 자식 노드 선택

    def simulate(self, state: Quoridor):  # 시뮬레이션: 게임 종료까지 진행
        current_simulation_state = state.clone()  # 상태 복사
        while current_simulation_state.check_winner() == 0:  # 승자가 나오지 않을 때
            # move = random.choice(current_simulation_state.get_movable_positions())  # 랜덤으로 이동
            move = self.brain.get_action(current_simulation_state.get_board(), current_simulation_state.get_movable_positions())  # AI가 추천한 좌표로 이동
            current_simulation_state.auto_turn(move_position=move)  # 이동
            reward = current_simulation_state.get_reward()  # 보상 계산
            if reward != 0:  # 보상이 0이 아닐 때
                return reward  # 보상 반환
        return 0

    def backpropagate(self, node: MCTSNode, reward):  # 역전파: 보상을 상위 노드로 전파
        while node:  # 노드가 존재할 때
            node.visits += 1  # 방문 횟수 증가
            node.value += reward  # 보상 증가
            node = node.parent  # 상위 노드로 이동
