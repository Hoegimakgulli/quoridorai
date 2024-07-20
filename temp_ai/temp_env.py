import numpy as np
import random


class character:
    def __init__(self, team: int, position: list[int], movable_positions: list[list[int]], attackable_positions: list[list[int]]):
        self.team = team
        self.position = position
        self.movable_positions = movable_positions
        self.attackable_positions = attackable_positions
        self.hp = 1

    def clone(self, position=None):
        if position is not None:
            return character(self.team, position, self.movable_positions)
        return character(self.team, self.position, self.movable_positions)


default_player = character(1, [4, 0], [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], [[1, 1], [-1, 1], [1, -1], [-1, -1]])
default_enemy = character(2, [4, 8], [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], [[1, 1], [-1, 1], [1, -1], [-1, -1]])


class Quoridor:
    def __init__(self):
        self.board = np.zeros((9, 9, 5), dtype=int)  ## 0 : 상, 1 : 하, 2 : 좌, 3: 우, 4: 오브젝트 위치
        self.players: list[character] = [default_player.clone()]
        self.enemys: list[character] = []
        self.set_enemy_positions(1)

    @property
    def players(self):
        return self._players

    @players.setter
    def players(self, value: list[list[int]]):
        self._players = value
        for player_position in self.players:
            self.board[player_position[0], player_position[1], 4] = 1

    @property
    def enemys(self):
        return self._enemys

    @enemys.setter
    def enemys(self, value: list[list[int]]):
        self._enemys = value
        for enemy_position in self.enemys:
            self.board[enemy_position[0], enemy_position[1], 4] = 2

    def clone(self):
        new_board = Quoridor()
        new_board.board = self.board.copy()
        new_board.players = [player.clone() for player in self.players]
        new_board.enemys = [enemy.clone() for enemy in self.enemys]
        return new_board

    def set_enemy_positions(self, count=1):
        random_positions = [[i, 8] for i in range(9)]
        for _ in range(count % 9):
            self.enemys.append(default_enemy.clone(random.choice([random_position for random_position in random_positions if random_position not in self.players])))
        if count > 9:
            raise ValueError("적의 수는 9명을 초과할 수 없습니다.")

    def move(self, index: int, position: list[int], team: int) -> bool:
        if self.can_move(index, position, team):
            self.board[self.players[index][0], self.players[index][1], 4] = 0
            self.players[index] = position
            self.board[self.players[index][0], self.players[index][1], 4] = team
            return True
        else:
            return False

    def can_move(self, index: int, new_position: list[int], team: int) -> bool:
        if team == 1:  # 플레이어
            pos = self.players[index].position
        else:  # 적
            pos = self.enemys[index].position

        diff = new_position - pos
        if diff in self.players[index].movable_positions:  # 이동 가능한 위치인지 확인
            return self.check_half_wall(pos, diff)
        else:
            return False

    def attack(self, index: int, position: list[int], team: int) -> bool:
        if self.can_attack(index, position, team):
            if team == 1:
                for enemy in self.enemys:
                    if enemy.position == position:
                        enemy.hp -= 1
                        break
            else:
                for player in self.players:
                    if player.position == position:
                        player.hp -= 1
                        break
            return True
        else:
            return False

    def can_attack(self, index: int, new_position: list[int], team: int) -> bool:
        if team == 1:
            pos = self.players[index].position
        else:
            pos = self.enemys[index].position

        diff = new_position - pos
        if diff in self.players[index].attackable_positions:
            return self.check_half_wall(pos, diff)
        else:
            return False

    def check_half_wall(self, position: list[int], direction: list[int]) -> bool:  # 벽이 반으로 막혀있는지 확인: 막혀있으면 False, 아니면 True
        if direction[0] != direction[1]:  # 대각선이 아닌 경우
            return True
        unit_diff = direction / abs(direction[0])
        for i in range(1, direction[0] + 1):
            check_pos = unit_diff * i + position
            pre_pos = unit_diff * (i - 1) + position
            if unit_diff[0] > 0:  # 우향
                current_unavailable = self.board[check_pos[0], check_pos[1], 2]  # 벽에 막히면 true, 아니면 false
                pre_unavailable = self.board[pre_pos[0], pre_pos[1], 3]
            else:  # 좌향
                current_unavailable = self.board[check_pos[0], check_pos[1], 3]
                pre_unavailable = self.board[pre_pos[0], pre_pos[1], 2]
            if unit_diff[1] > 0:  # 상향
                current_unavailable = current_unavailable and self.board[check_pos[0], check_pos[1], 1]  # 벽에 의해 둘 다 막히면 true, 아니면 false
                pre_unavailable = pre_unavailable and self.board[pre_pos[0], pre_pos[1], 0]
            else:  # 하향
                current_unavailable = current_unavailable and self.board[check_pos[0], check_pos[1], 0]
                pre_unavailable = pre_unavailable and self.board[pre_pos[0], pre_pos[1], 1]
            if current_unavailable or pre_unavailable:
                return False
        return True

    def check_winner(self):
        if len([player for player in self.players if player.hp > 0]) == 0:
            return 2
        elif len([enemy for enemy in self.enemys if enemy.hp > 0]) == 0:
            return 1
        else:
            return 0

    def get_reward(self):
        if self.check_winner() == 1:
            return 1
        elif self.check_winner() == 2:
            return -1
        else:
            return 0

    def get_legal_moves(self, team: int, index: int) -> list[list[int]]:
        if team == 1:
            return self.players[index].movable_positions
        else:
            return self.enemys[index].movable_positions
