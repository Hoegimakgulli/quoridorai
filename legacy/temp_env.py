import numpy as np
import random
import sys
import os
import torch
from enum import Enum
from typing import List, Union

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from enemy_character import path_finding


class Point:
    x: int
    y: int

    def __init__(self, *args):
        if len(args) == 0:
            self.x = 0
            self.y = 0
        elif len(args) == 1:
            if type(args[0]) == list:
                self.x = args[0][0]
                self.y = args[0][1]
            elif type(args[0]) == Point:
                self.x = args[0].x
                self.y = args[0].y
        elif len(args) == 2:
            self.x = args[0]
            self.y = args[1]

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def __repr__(self):
        return "Point({}, {})".format(self.x, self.y)

    def __add__(self, other):
        return Point(int(self.x + other.x), int(self.y + other.y))

    def __sub__(self, other):
        return Point(int(self.x - other.x), int(self.y - other.y))

    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            return Point(int(self.x * other), int(self.y * other))

    def __truediv__(self, other):
        if type(other) == int or type(other) == float:
            return Point(int(self.x / other), int(self.y / other))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("list index out of range")

    def __list__(self):
        return [self.x, self.y]

    def set_list(_list: list[list[int]]):
        new_list: list[Point] = []
        for i in _list:
            new_list.append(Point(i))
        return new_list


class Character:
    def __init__(self, team: int, position: Point, movable_positions: list[Point], attackable_positions: list[Point], maxHp: int, atk: int, move_ctrl: int, increase_move_ctrl: int):
        self.team = team
        self.position: Point = position
        self.movable_positions: list[Point] = movable_positions
        self.attackable_positions: list[Point] = attackable_positions
        self.maxHp = maxHp
        self.hp = maxHp
        self.atk = atk
        self.move_ctrl = move_ctrl
        self.increase_move_ctrl = increase_move_ctrl

    def clone(self, position=None):
        if position is not None:
            return Character(self.team, position, self.movable_positions, self.attackable_positions, self.maxHp, self.atk, self.move_ctrl, self.increase_move_ctrl)
        return Character(self.team, self.position, self.movable_positions, self.attackable_positions, self.maxHp, self.atk, self.move_ctrl, self.increase_move_ctrl)

    def get_abs_movable_positions(self):
        return [self.position + i for i in self.movable_positions]

    def heal_move_ctrl(self):
        self.move_ctrl = min(self.move_ctrl + self.increase_move_ctrl, 100)


default_player = Character(1, Point(4, 0), Point.set_list([[1, 0], [-1, 0], [0, 1], [0, -1]]), Point.set_list([[1, 0], [-1, 0], [0, 1], [0, -1]]), 1, 1, 100, 34)
default_enemy = Character(2, Point(4, 8), Point.set_list([[1, 0], [-1, 0], [0, 1], [0, -1]]), Point.set_list([[1, 0], [-1, 0], [0, 1], [0, -1]]), 1, 1, 100, 34)

class Wall:
    class EDirection(Enum):
        Vertical = 0
        Horizontal = 1
    def __init__(self, position: Point, direction: EDirection):
        self.position : Point = position
        self.direction : Wall.EDirection = direction


class Quoridor:
    def __init__(self):
        self.board = np.zeros((5, 9, 9), dtype=int)  ## 0 : 상, 1 : 하, 2 : 좌, 3: 우, 4: 오브젝트 위치
        self.players: list[Character] = [default_player.clone()]
        self.enemys: list[Character] = []
        self.walls
        self.set_enemy_positions(1)

    @property
    def players(self):
        return self._players

    @players.setter
    def players(self, value: list[list[int]]):
        self._players = value
        # for player in self.players:
        # self.board[4, player.position.x, player.position.y] = 1

    @property
    def enemys(self):
        return self._enemys

    @enemys.setter
    def enemys(self, value: list[list[int]]):
        self._enemys = value
        # for enemy in self.enemys:
        # self.board[4, enemy.position.x, enemy.position.y] = 2

    def clone(self):
        new_board = Quoridor()
        new_board.board = self.board.copy()
        new_board.players = [player.clone() for player in self.players]
        new_board.enemys = [enemy.clone() for enemy in self.enemys]
        return new_board

    def reset(self):
        self.board = np.zeros((5, 9, 9), dtype=int)
        self.players = [default_player.clone()]
        self.enemys = []
        self.set_enemy_positions(1)

    def get_board(self):
        return torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)

    def set_enemy_positions(self, count=1):
        random_positions = [[i, 8] for i in range(9)]
        for _ in range(count % 9):
            self.enemys.append(default_enemy.clone(Point(random.choice([random_position for random_position in random_positions if random_position not in [enemy.position for enemy in self.enemys]]))))
        for pos in [enemy.position for enemy in self.enemys]:
            self.board[4, pos.x, pos.y] = 2
        if count > 9:
            raise ValueError("적의 수는 9명을 초과할 수 없습니다.")

    def move(self, index: int, position: Point, team: int) -> bool:
        position = Point(position)
        if self.can_move(index, position, team):
            if team == 1:
                self.board[4, self.players[index].position[0], self.players[index].position[1]] = 0
                self.players[index].position = position
                self.board[4, self.players[index].position[0], self.players[index].position[1]] = team
            else:
                self.board[4, self.enemys[index].position[0], self.enemys[index].position[1]] = 0
                self.enemys[index].position = position
                self.board[4, self.enemys[index].position[0], self.enemys[index].position[1]] = team
            return True
        else:
            return False

    def move_clone(self, index: int, position: Point, team: int):
        quoridor_clone = self.clone()
        position = Point(position)
        if self.can_move(index, position, team):
            if team == 1:
                quoridor_clone.board[4, self.players[index].position[0], self.players[index].position[1]] = 0
                quoridor_clone.players[index].position = position
                quoridor_clone.board[4, self.players[index].position[0], self.players[index].position[1]] = team
            else:
                quoridor_clone.board[4, self.enemys[index].position[0], self.enemys[index].position[1]] = 0
                quoridor_clone.enemys[index].position = position
                quoridor_clone.board[4, self.enemys[index].position[0], self.enemys[index].position[1]] = team
            return quoridor_clone
        else:
            return quoridor_clone

    def move_debug(self, index: int, position: Point, team: int):
        position = Point(position)
        if team == 1:
            self.board[4, self.players[index].position[0], self.players[index].position[1]] = 0
            self.players[index].position = position
            self.board[4, self.players[index].position[0], self.players[index].position[1]] = team
        else:
            self.board[4, self.enemys[index].position[0], self.enemys[index].position[1]] = 0
            self.enemys[index].position = position
            self.board[4, self.enemys[index].position[0], self.enemys[index].position[1]] = team

    def can_move(self, index: int, new_position: Point, team: int) -> bool:
        if team == 1:  # 플레이어
            pos = self.players[index].position
        else:  # 적
            pos = self.enemys[index].position
        new_position = Point(new_position)
        diff = new_position - pos
        if diff in (self.players[index].movable_positions if team == 1 else self.enemys[index].movable_positions):  # 이동 가능한 위치인지 확인
            return self.check_half_wall(pos, diff)
        else:
            return False

    def attack(self, index: int, team: int, show_log: bool = False) -> bool:
        attackable_positions = self.players[index].attackable_positions if team == 1 else self.enemys[index].attackable_positions
        for position in attackable_positions:
            if self.can_attack(index, position, team):
                if team == 1:
                    for enemy in self.enemys:
                        # print(enemy.position, self.players[index].position, position, position + self.players[index].position)
                        if enemy.position == position + self.players[index].position:
                            enemy.hp -= 1
                            if show_log:
                                print("enemy {0}'s hp: {1}".format(self.enemys.index(enemy), enemy.hp))
                            if enemy.hp <= 0:
                                if show_log:
                                    print("enemy {0} died".format(self.enemys.index(enemy)))
                                self.board[4, enemy.position[0], enemy.position[1]] = 0
                            return True
                else:
                    for player in self.players:
                        if player.position == position + self.enemys[index].position:
                            player.hp -= 1
                            if show_log:
                                print("player {0}'s hp: {1}".format(self.players.index(player), player.hp))
                            if player.hp <= 0:
                                if show_log:
                                    print("player {0} died".format(self.players.index(player)))
                                self.board[4, player.position[0], player.position[1]] = 0
                            return True
        return False

    def can_attack(self, index: int, attack_position: list[int], team: int) -> bool:
        if team == 1:
            pos = self.players[index].position
        else:
            pos = self.enemys[index].position
        diff = Point(attack_position)
        # print(diff)
        if diff in (self.players[index].attackable_positions if team == 1 else self.enemys[index].attackable_positions):
            # print("checked")
            return self.check_half_wall(pos, diff)
        else:
            return False

    def set_wall(self, position: Point, direction: int) -> bool:
        position = Point(position)
        if self.can_set_wall(position, direction):
    
    def can_set_wall(self, position: Point, direction: int) -> bool:
        #겹치는 지 확인


    def check_half_wall(self, position: list[int], direction: list[int]) -> bool:  # 벽이 반으로 막혀있는지 확인: 막혀있으면 False, 아니면 True
        if direction[0] != direction[1]:  # 대각선이 아닌 경우
            return True
        # unit_diff = [direction[0] / abs(direction[0]), direction[1] / abs(direction[1])]
        unit_diff: Point = Point(direction) / abs(direction[0])
        position = Point(position)
        for i in range(1, direction[0] + 1):
            check_pos: Point = unit_diff * i + position
            pre_pos: Point = unit_diff * (i - 1) + position
            # print(check_pos)
            if unit_diff[0] > 0:  # 우향
                current_unavailable = self.board[2, int(check_pos[0]), int(check_pos[1])]  # 벽에 막히면 true, 아니면 false
                pre_unavailable = self.board[3, int(pre_pos[0]), int(pre_pos[1])]
            else:  # 좌향
                current_unavailable = self.board[3, int(check_pos[0]), int(check_pos[1])]
                pre_unavailable = self.board[2, int(pre_pos[0]), int(pre_pos[1])]
            if unit_diff[1] > 0:  # 상향
                current_unavailable = current_unavailable and self.board[1, int(check_pos[0]), int(check_pos[1])]  # 벽에 의해 둘 다 막히면 true, 아니면 false
                pre_unavailable = pre_unavailable and self.board[0, int(pre_pos[0]), int(pre_pos[1])]
            else:  # 하향
                current_unavailable = current_unavailable and self.board[0, int(check_pos[0]), int(check_pos[1])]
                pre_unavailable = pre_unavailable and self.board[1, int(pre_pos[0]), int(pre_pos[1])]
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

    def get_legal_moves(self, team: int, index: int) -> list[Point]:
        if team == 1:
            return [
                pos
                for pos in self.players[index].get_abs_movable_positions()
                if (pos.x >= 0 and pos.x < 9) and (pos.y >= 0 and pos.y < 9) and not pos in [enemy.position for enemy in self.enemys] and not pos in [player.position for player in self.players]
            ]
        else:
            return [
                pos
                for pos in self.enemys[index].get_abs_movable_positions()
                if (pos.x >= 0 and pos.x < 9) and (pos.y >= 0 and pos.y < 9) and not pos in [enemy.position for enemy in self.enemys] and not pos in [player.position for player in self.players]
            ]

    def enemy_turn(self, index: int = -1, do_move: bool = True):
        ## 어떤 적이 움직이게 할지 결정
        if index == -1:
            index = random.randint(0, len(self.enemys) - 1)
        enemy = self.enemys[index]
        if enemy.hp > 0:
            enemy_path_list: list[list[int]] = []
            enemy_path_len_list: list[int] = []
            for player in self.players:
                if player.hp > 0:
                    # print(f"player: {player.position}, enemy: {enemy.position}")
                    enemy_path_list.append(path_finding(list(player.position), list(enemy.position), 9, 9, board_to_wall_data(self.board)))
                    enemy_path_list[-1].append(list(player.position))
                    # print(f"path: {enemy_path_list[-1]}")
                    enemy_path_len_list.append(len(enemy_path_list[-1]))
            path = enemy_path_list[enemy_path_len_list.index(min(enemy_path_len_list))]
            # print(path)
            if do_move:
                self.move(self.enemys.index(enemy), path[1], 2)
                self.attack(self.enemys.index(enemy), 2)
            return path[1], index

    def print_board(self, index):
        print(self.board[index])


def board_to_wall_data(board: np.ndarray) -> list[list[list[int]]]:
    wall_data = []
    empty_list = np.zeros(81, dtype=int).tolist()
    for i in range(8):
        if i % 2 == 0:
            wall_data.append(board[i // 2, :, :].reshape(81).tolist())
        else:
            wall_data.append(empty_list)
    wall_data = np.array(wall_data).T.tolist()
    return wall_data


if __name__ == "__main__":
    env: Quoridor = Quoridor()
    env.reset()
    env.move_debug(0, Point(4, 2), 2)
    env.print_board(4)
    env.move(0, Point(4, 1), 1)
    env.print_board(4)
    print(env.attack(0, 1))
    env.print_board(4)
    print(env.check_winner())
    # while env.check_winner() == 0:
    #     env.move(0, Point(list(map(int, input("player move: ").split(",")))), 1)
    #     env.attack(0, 1)
    #     env.print_board(4)
    #     print(env.enemy_turn())
    #     env.print_board(4)
