import numpy as np
import random
from ai.utils import Vector2
from ai.environment import Quoridor
from ai.model import DQNAgent
import math

##TODO: AGENT 개발 후 추가


class MCTSNode:
    def __init__(self, state: Quoridor, move: Vector2, weight: float, parent=None):
        self.state: Quoridor = state
        self.move: Vector2 = move
        self.weight: float = weight
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        movable_positions = self.state.get_movable_positions()
        if movable_positions is None or len(movable_positions) == 0:
            return True
        return len(self.children) == len(self.state.get_movable_positions())

    def best_child(self, exploration_weight=1.4):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                choices_weights.append(float("inf"))  # 탐험하지 않은 노드는 우선적으로 선택
            else:
                uct_value = (child.value / child.visits) + exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits)
                choices_weights.append(uct_value)
        return self.children[choices_weights.index(max(choices_weights))]


class MCTS:
    def __init__(self, game: Quoridor, brain: DQNAgent, epsilon, simulations=None):
        self.game: Quoridor = game
        self.brain: DQNAgent = brain
        self.epsilon = epsilon
        if simulations is None:
            simulations = self.sigmoid_simulate_count(np.var([weight for action, weight in self.brain.get_action(self.game.get_board(), self.game.get_movable_positions(), self.epsilon)]))
        self.simulations = simulations
        self.convergence_threshold = 0.95
        self.ucb_threshold = 0.1

        self.real_simulation_count = 0

    def search(self) -> MCTSNode:  # 탐색
        root = MCTSNode(self.game.clone(), Vector2(4, 0), 0)

        for _ in range(self.simulations):
            node = self.select(root)
            reward = self.simulate(node.state)
            self.backpropagate(node, reward)

            self.real_simulation_count += 1

            if self.real_simulation_count > 50:
                if self.check_convergence(root):
                    break

                if self.check_ucb_difference(root):
                    break

        return root.best_child(exploration_weight=0)

    def select(self, node: MCTSNode):  # 선택: 가장 유망한 노드 선택
        while node.state.check_winner() == 0 and node.is_fully_expanded() and len(node.children) > 0:  # 노드가 완전 확장되지 않았거나, 승자가 나오지 않았을 때
            node = node.best_child()  # 가장 유망한 노드 선택
        if node.state.check_winner() == 0 and not node.is_fully_expanded():  # 승자가 나오지 않았고, 노드가 완전 확장되지 않았을 때
            return self.expand(node)  # 확장
        return node

    def expand(self, node: MCTSNode):  # 확장: 노드 확장
        moves = sorted(self.brain.get_action(node.state.get_board(), node.state.get_movable_positions(), self.epsilon), key=lambda x: x[1], reverse=True)  # AI가 추천한 좌표
        if len(moves) == 0:
            return node
        # print(f"to_move:{moves}, available:{node.state.get_movable_positions()} - MCTS")
        for move, weight in moves:  # 이동 가능한 위치에 대해
            new_state = node.state.clone()  # 상태 복사
            new_state.auto_turn(move_position=move)  # 턴 진행
            child_node = MCTSNode(new_state, move, weight, parent=node)  # 자식 노드 생성
            node.children.append(child_node)  # 자식 노드 추가
        return node.children[round(np.abs(np.random.normal(0, 1)) ** 2 * len(node.children)) % len(node.children)]  # 랜덤으로 자식 노드 선택

    def simulate(self, state: Quoridor):  # 시뮬레이션: 게임 종료까지 진행
        current_simulation_state = state.clone()  # 상태 복사
        while current_simulation_state.check_winner() == 0 or current_simulation_state.turn < 100:  # 승자가 나오지 않을 때
            # move = random.choice(current_simulation_state.get_movable_positions())  # 랜덤으로 이동
            actions = self.brain.get_action(current_simulation_state.get_board(), current_simulation_state.get_movable_positions(), self.epsilon)
            if len(actions) != 0:
                # move, weight = max(actions, key=lambda x: x[1])  # AI가 추천한 좌표로 이동
                move = actions[round(np.abs(np.random.normal(0, 1)) ** 2 * len(actions)) % len(actions)][0]
            else:
                move = None
            current_simulation_state.auto_turn(move_position=move)  # 이동
            reward = current_simulation_state.reward()  # 보상 계산
            if reward != 0:  # 보상이 0이 아닐 때
                return reward  # 보상 반환
        return 0

    def backpropagate(self, node: MCTSNode, reward):  # 역전파: 보상을 상위 노드로 전파
        while node:  # 노드가 존재할 때
            node.visits += 1  # 방문 횟수 증가
            node.value += reward  # 보상 증가
            node = node.parent  # 상위 노드로 이동

    def check_convergence(self, node: MCTSNode) -> bool:
        """노드 방문 횟수를 기반으로 탐색 수렴 여부를 판단"""
        if len(node.children) == 0:
            return False

        visit_counts = [child.visits for child in node.children]
        total_visits = sum(visit_counts)
        max_visits = max(visit_counts)

        # 가장 많이 방문된 자식 노드가 임계값 이상일 경우 조기 종료
        return (max_visits / total_visits) >= self.convergence_threshold

    def check_ucb_difference(self, node: MCTSNode) -> bool:
        """UCT 값의 차이를 기준으로 조기 종료 여부를 판단"""
        if len(node.children) == 0:
            return False

        ucb_values = []
        for child in node.children:
            if child.visits == 0:
                ucb_values.append(float("inf"))
            else:
                ucb_value = (child.value / child.visits) + math.sqrt(2 * math.log(node.visits) / child.visits)
                ucb_values.append(ucb_value)

        max_ucb = max(ucb_values)
        for value in ucb_values:
            if abs(max_ucb - value) < self.ucb_threshold:
                return False  # UCT 값 차이가 임계값보다 작으면 계속 탐색

        return True  # UCT 값 차이가 충분히 벌어졌으면 종료

    def sigmoid_simulate_count(self, variance, min=50, max=150, k=2.5):

        def sigmoid(x):
            return 1 / (1 + np.exp(-k * x))

        scale = sigmoid(1 - variance)

        result = int(min + scale * (max - min))
        if result < min:
            return min
        return result
