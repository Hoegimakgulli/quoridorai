import numpy as np
from ai.utils import Vector2
import torch
import random


class Character:
    def __init__(self, team: int, position: Vector2, movable_positions: list[Vector2], attackable_positions: list[Vector2], maxHp: int, atk: int, move_ctrl: int, increase_move_ctrl: int):
        self.team = team
        self.position = position
        self.movable_positions = movable_positions
        self.attackable_positions = attackable_positions
        self.maxHp = maxHp
        self.hp = maxHp
        self.atk = atk
        self.move_ctrl = move_ctrl
        self.increase_move_ctrl = increase_move_ctrl

        self.is_active = True

    def clone(self, position=None):
        if position is not None:
            return Character(self.team, position, self.movable_positions, self.attackable_positions, self.maxHp, self.atk, self.move_ctrl, self.increase_move_ctrl)
        return Character(self.team, self.position, self.movable_positions, self.attackable_positions, self.maxHp, self.atk, self.move_ctrl, self.increase_move_ctrl)

    def get_abs_movable_positions(self):
        return [self.position + i for i in self.movable_positions]

    def get_abs_attackable_positions(self):
        return [self.position + i for i in self.attackable_positions]

    def update(self):
        if not self.is_active:
            return
        self.move_ctrl = min(self.move_ctrl + self.increase_move_ctrl, 100)

    def attack(self, target):
        return target.damage(self.atk)

    def damage(self, atk):
        self.hp -= atk
        if self.hp <= 0:
            self.die()
            return True
        return False

    def die(self):
        self.is_active = False


class Wall:
    def __init__(self, position: Vector2, is_horizontal: bool):
        self.position = position
        self.is_horizontal = is_horizontal


default_player = Character(1, Vector2(4, 0), [Vector2.up(), Vector2.down(), Vector2.right(), Vector2.left()], [Vector2.up(), Vector2.down(), Vector2.right(), Vector2.left()], 1, 1, 100, 34)
default_enemy = Character(2, Vector2(4, 8), [Vector2.up(), Vector2.down(), Vector2.right(), Vector2.left()], [Vector2.up(), Vector2.down(), Vector2.right(), Vector2.left()], 1, 1, 100, 34)


class Quoridor:
    def __init__(self):
        self.board = np.zeros((5, 9, 9), dtype=int)  ## 0 : 상, 1 : 하, 2 : 좌, 3: 우, 4: 오브젝트 위치
        self.player: Character = default_player.clone()
        self.enemy_list: list[Character] = []
        self.add_enemy(random.randint(1, 8))
        self.wall_list: list[Wall] = []
        self.turn = 0
        self.recalculate_board()

    @property
    def character_list(self):
        return [self.player] + self.enemy_list

    @property
    def current_character(self):
        if self.turn % 2 == 0:
            return self.player
        else:
            return self.enemy_list[(self.turn // 2) % len(self.enemy_list)]

    def clone(self):
        q = Quoridor()
        q.board = self.board.copy()
        q.player = self.player.clone()
        q.enemy_list = [i.clone() for i in self.enemy_list]
        q.wall_list = [Wall(i.position, i.is_horizontal) for i in self.wall_list]
        q.turn = self.turn
        q.recalculate_board()
        return q

    def reset(self):
        self.__init__()

    def get_board(self):
        return torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)

    def recalculate_board(self):
        self.board = np.zeros((5, 9, 9), dtype=int)  ## 0 : 상, 1 : 하, 2 : 좌, 3: 우, 4: 오브젝트 위치
        for character in self.character_list:
            if character.is_active:
                self.board[4, int(character.position.x), int(character.position.y)] = character.team
        for wall in self.wall_list:
            related_pos_list = []

            for diff in [Vector2.zero(), Vector2.right(), Vector2.up(), Vector2(1, 1)]:
                if (wall.position + diff).x < 0 or (wall.position + diff).y < 0 or (wall.position + diff).x >= 9 or (wall.position + diff).y >= 9:
                    related_pos_list.append(None)
                    continue
                related_pos_list.append(wall.position + diff)

            for i, related_pos in enumerate(related_pos_list):
                if related_pos is not None:
                    if wall.is_horizontal:
                        if i == 0 or i == 1:
                            self.board[0, int(related_pos.x), int(related_pos.y)] = 1
                        elif i == 2 or i == 3:
                            self.board[1, int(related_pos.x), int(related_pos.y)] = 1
                    else:
                        if i == 0 or i == 2:
                            self.board[3, int(related_pos.x), int(related_pos.y)] = 1
                        elif i == 1 or i == 3:
                            self.board[2, int(related_pos.x), int(related_pos.y)] = 1

    def add_enemy(self, count: int):
        if count < 0 or count > 8:
            raise ValueError("Invalid count")
        for i in range(count):
            new_enemy = default_enemy.clone(Vector2(random.randint(0, 8), 8))
            self.enemy_list.append(new_enemy)

    def move(self, move_position: Vector2, character: Character):
        # print(f"to_move:{move_position}, available:{character.get_abs_movable_positions()}")
        if not self.can_move(move_position, character):
            return False
        character.position = move_position
        self.recalculate_board()
        return True

    def can_move(self, move_position: Vector2, character: Character):
        if type(move_position) is not Vector2:
            raise ValueError("Invalid move position")
        if move_position.x < 0 or move_position.y < 0 or move_position.x >= 9 or move_position.y >= 9:
            return False
        if move_position not in character.get_abs_movable_positions():
            return False
        if self.board[4, int(move_position.x), int(move_position.y)] != 0:
            return False
        diff = move_position - character.position
        return self.check_wall(character.position, diff)

    def attack(self, character: Character):
        if character.team == 1:  # 플레이어가 공격하는 경우
            nearest_enemy = None
            nearest_distance = None
            for enemy in self.enemy_list:
                if self.can_attack(character, enemy):
                    if nearest_enemy is None or character.position.distance(enemy.position) < nearest_distance:
                        nearest_enemy = enemy
                        nearest_distance = character.position.distance(enemy.position)
            if nearest_enemy is not None:
                return character.attack(nearest_enemy)
        elif character.team == 2:  # 적이 공격하는 경우
            if self.can_attack(character, self.player):
                return character.attack(self.player)

    def can_attack(self, character: Character, target: Character):
        if character.team == target.team:
            return False
        if character.position not in target.get_abs_attackable_positions():
            return False
        return self.check_wall(character.position, target.position - character.position)

    def next_turn(self):
        self.turn += 1
        for char in self.character_list:
            char.update()
        self.recalculate_board()

    def auto_turn(self, character: Character = None, move_position: Vector2 = None):
        if character is None:
            character = self.current_character  # TODO: 현재 턴에 맞는 캐릭터 선택
        if move_position is not None:
            if not self.move(move_position, character):
                raise Exception("Invalid move")
        self.attack(character)
        self.next_turn()

    def get_movable_positions(self, character: Character = None):
        if character is None:
            character = self.current_character
        movable_positions = [i for i in character.get_abs_movable_positions() if self.can_move(i, character)]
        if len(movable_positions) == 0:
            return None
        return movable_positions

    def check_wall(self, position: Vector2, vector: Vector2):  # 벽 검사: 벽이 있으면 False, 없으면 True
        if abs(vector.x) > abs(vector.y):
            if vector.x > 0:
                return self.board[0, int(position.x), int(position.y)] == 0
            else:
                return self.board[1, int(position.x), int(position.y)] == 0
        elif abs(vector.x) < abs(vector.y):
            if vector.y > 0:
                return self.board[3, int(position.x), int(position.y)] == 0
            else:
                return self.board[2, int(position.x), int(position.y)] == 0
        else:
            return self.check_half_wall(position, vector)

    def check_half_wall(self, position: Vector2, vector: Vector2):  # 벽이 반으로 막혀있는지 확인: 막혀있으면 False, 아니면 True
        if abs(vector.x) != abs(vector.y):  # 대각선이 아닌 경우
            return True
        # unit_diff = [direction[0] / abs(direction[0]), direction[1] / abs(direction[1])]
        unit_diff: Vector2 = vector.normalize()
        for i in range(1, vector.x + 1):
            check_pos: Vector2 = unit_diff * i + position
            pre_pos: Vector2 = unit_diff * (i - 1) + position
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
        if not self.player.is_active:
            return 2
        if not any([i.is_active for i in self.enemy_list]):
            return 1
        return 0

    def reward(self):
        winner = self.check_winner()
        if winner == 1:
            return 1
        elif winner == 2:
            return -1
        return 0

    def print_board(self, index):
        print(self.board[index])
