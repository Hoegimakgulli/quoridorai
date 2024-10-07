import numpy as np
from ai.utils import Vector2, CircularQueue
import torch
import random


class Character:  # 캐릭터 클래스
    def __init__(self, team: int, position: Vector2, movable_positions: list[Vector2], attackable_positions: list[Vector2], maxHp: int, atk: int, move_ctrl: int, increase_move_ctrl: int):  # 생성자
        self.team = team  # 팀
        self.position = position  # 위치
        self.movable_positions = movable_positions  # 이동 가능한 위치
        self.attackable_positions = attackable_positions  # 공격 가능한 위치
        self.maxHp = maxHp  # 최대 체력
        self.hp = maxHp  # 현재 체력
        self.atk = atk  # 공격력
        self.move_ctrl = move_ctrl  # 이동력
        self.increase_move_ctrl = increase_move_ctrl  # 이동력 증가량

        self.is_active = True  # 활성화 여부

    def clone(self, position=None):  # 복제
        if position is not None:  # 위치가 주어진 경우
            new_character = Character(
                self.team, position, self.movable_positions, self.attackable_positions, self.maxHp, self.atk, self.move_ctrl, self.increase_move_ctrl
            )  # 새로운 캐릭터 (좌표 포함) 생성
        else:
            new_character = Character(self.team, self.position, self.movable_positions, self.attackable_positions, self.maxHp, self.atk, self.move_ctrl, self.increase_move_ctrl)  # 새로운 캐릭터 생성
        new_character.hp = self.hp  # 체력 복사
        new_character.is_active = self.is_active
        return new_character  # 새로운 캐릭터 반환

    def get_abs_movable_positions(self):  # 절대 이동 좌표 반환
        return [self.position + i for i in self.movable_positions]  # 이동 가능한 위치 + 현재 위치 = 절대 좌표 반환

    def get_abs_attackable_positions(self):  # 절대 공격 좌표 반환
        return [self.position + i for i in self.attackable_positions]  # 공격 가능한 위치 + 현재 위치 = 절대 좌표 반환

    def update(self):  # 업데이트
        if not self.is_active:  # 활성화되지 않은 경우
            return  # 종료
        self.move_ctrl = min(self.move_ctrl + self.increase_move_ctrl, 100)  # 이동력 증가

    def attack(self, target):  # 공격
        return target.damage(self.atk)  # 대상에게 공격력만큼 데미지를 입힘

    def damage(self, atk):  # 데미지
        self.hp -= atk  # 체력 감소
        if self.hp <= 0:  # 체력이 0 이하인 경우
            self.die()  # 사망
            return True  # True 반환
        return False  # False 반환

    def die(self):  # 사망
        self.is_active = False  # 활성화 여부를 False로 변경


class Wall:  # 벽 클래스
    def __init__(self, position: Vector2, is_horizontal: bool):  # 생성자
        self.position = position  # 위치
        self.is_horizontal = is_horizontal  # 수평 여부


default_player = Character(
    1,
    Vector2(4, 0),
    [Vector2.up(), Vector2.down(), Vector2.right(), Vector2.left()],
    [Vector2.up(), Vector2.down(), Vector2.right(), Vector2.left(), Vector2.up() * 2, Vector2.down() * 2, Vector2.right() * 2, Vector2.left() * 2],
    1,
    1,
    100,
    34,
)  # 기본 플레이어
default_enemy = Character(2, Vector2(4, 8), [Vector2.up(), Vector2.down(), Vector2.right(), Vector2.left()], [Vector2.up(), Vector2.down(), Vector2.right(), Vector2.left()], 1, 1, 100, 34)  # 기본 적


class Quoridor:  # 쿼리도 클래스
    def __init__(self):  # 생성자
        self.board = np.zeros((5, 9, 9), dtype=int)  ## 0 : 상, 1 : 하, 2 : 좌, 3: 우, 4: 오브젝트 위치
        self.player: Character = default_player.clone()  # 플레이어
        self.enemy_list: list[Character] = []  # 적 리스트
        self.add_enemy(random.randint(1, 8))  # 적 추가
        self.enemy_circle_queue: CircularQueue = CircularQueue(len(self.enemy_list))  # 적 순환 큐
        self.enemy_circle_queue.enqueue_list(self.enemy_list)  # 적 순환 큐에 적 리스트 추가
        self.wall_list: list[Wall] = []  # 벽 리스트
        self.turn = 0  # 턴
        self.recalculate_board()  # 보드 재계산

    @property
    def character_list(self):  # 캐릭터 리스트
        return [self.player] + self.enemy_list  # 플레이어 + 적 리스트 반환

    @property
    def current_character(self):  # 현재 캐릭터
        if self.turn % 2 == 0:  # 턴이 짝수인 경우
            return self.player  # 플레이어 반환
        else:  # 홀수인 경우
            if len(self.enemy_list) == 1:  # 적이 하나인 경우
                return self.enemy_list[0]  # 적 반환
            if not any([i.is_active for i in self.enemy_list]):  # 적이 모두 죽은 경우
                return self.player  # 플레이어 반환
            while True:  # 적이 활성화되지 않은 경우
                enemy = self.enemy_circle_queue.peek()  # 적 반환 (지금은 생성된 순서대로 턴 돌아감)
                if enemy is not None and enemy.is_active:
                    return enemy  # 적 반환
                else:
                    self.enemy_circle_queue.forward()  # 순환 큐에서 다음 적으로 이동

    def clone(self):  # 복제
        q = Quoridor()  # 새로운 쿼리도 생성
        q.board = self.board.copy()  # 보드 복사
        q.player = self.player.clone()  # 플레이어 복제
        q.enemy_list = [i.clone() for i in self.enemy_list]  # 적 리스트 복제
        q.enemy_circle_queue = CircularQueue(len(q.enemy_list))  # 적 순환 큐 생성
        for i in range(len(q.enemy_list) + 1):
            if self.enemy_circle_queue.queue[i] is None:
                continue
            q.enemy_circle_queue.queue[i] = q.enemy_list[self.enemy_list.index(self.enemy_circle_queue.queue[i])]
        q.enemy_circle_queue.front = self.enemy_circle_queue.front  # 적 순환 큐 복제
        q.enemy_circle_queue.rear = self.enemy_circle_queue.rear
        q.wall_list = [Wall(i.position, i.is_horizontal) for i in self.wall_list]  # 벽 리스트 복제
        q.turn = self.turn  # 턴 복사
        q.recalculate_board()  # 보드 재계산
        return q  # 새로운 쿼리도 반환

    def reset(self):  # 리셋
        self.__init__()  # 생성자 호출

    def get_board(self):  # 보드 반환
        return torch.tensor(self.board, dtype=torch.float32).unsqueeze(0)  # 보드를 텐서로 변환하여 반환

    def recalculate_board(self):  # 보드 재계산
        self.board = np.zeros((5, 9, 9), dtype=int)  ## 0 : 상, 1 : 하, 2 : 좌, 3: 우, 4: 오브젝트 위치
        for character in self.character_list:  # 캐릭터 리스트에 대해 반복
            if character.is_active:  # 활성화된 경우(살아있는 경우)
                self.board[4, int(character.position.x), int(character.position.y)] = character.team  # 캐릭터 위치에 팀 번호 저장
        for wall in self.wall_list:  # 벽 리스트에 대해 반복
            relative_pos_list = []  # 상대 위치 리스트

            for diff in [Vector2.zero(), Vector2.right(), Vector2.up(), Vector2(1, 1)]:  # 상대 위치 계산
                if (wall.position + diff).x < 0 or (wall.position + diff).y < 0 or (wall.position + diff).x >= 9 or (wall.position + diff).y >= 9:  # 범위를 벗어나는 경우
                    relative_pos_list.append(None)  # None 추가 (because: list의 길이를 4개로 유지하기 위함)
                    continue
                relative_pos_list.append(wall.position + diff)  # 상대 위치 추가

            for i, related_pos in enumerate(relative_pos_list):  # 상대 위치에 대해 반복
                if related_pos is not None:  # None이 아닌 경우
                    if wall.is_horizontal:  # 수평인 경우
                        if i == 0 or i == 1:  # 아래 2개
                            self.board[0, int(related_pos.x), int(related_pos.y)] = 1  # 상에 벽이 있음을 표시
                        elif i == 2 or i == 3:  # 위에 2개
                            self.board[1, int(related_pos.x), int(related_pos.y)] = 1  # 하에 벽이 있음을 표시
                    else:  # 수직인 경우
                        if i == 0 or i == 2:  # 왼쪽 2개
                            self.board[3, int(related_pos.x), int(related_pos.y)] = 1  # 우에 벽이 있음을 표시
                        elif i == 1 or i == 3:  # 오른쪽 2개
                            self.board[2, int(related_pos.x), int(related_pos.y)] = 1  # 좌에 벽이 있음을 표시

    def add_enemy(self, count: int):  # 적 추가
        if count < 0 or count > 8:  # 0 미만 또는 8 초과인 경우
            raise ValueError("Invalid count")  # 오류 발생
        for i in range(count):  # count만큼 반복
            new_pos = Vector2(random.randint(0, 8), 8)  # 새로운 위치
            while new_pos in [i.position for i in self.character_list]:  # 위치가 겹치지 않을 때까지 반복
                new_pos = Vector2(random.randint(0, 8), 8)
            new_enemy = default_enemy.clone(Vector2(random.randint(0, 8), 8))  # 새로운 적 생성
            self.enemy_list.append(new_enemy)  # 적 리스트에 추가

    def move(self, move_position: Vector2, character: Character):  # 이동
        # print(f"to_move:{move_position}, available:{character.get_abs_movable_positions()}")
        if not self.can_move(move_position, character):  # 이동할 수 없는 경우
            return False  # False 반환
        character.position = move_position  # 캐릭터 위치 이동
        self.recalculate_board()  # 보드 재계산
        return True  # True 반환

    def can_move(self, move_position: Vector2, character: Character):
        if type(move_position) is not Vector2:  # 이동할 위치가 벡터가 아닌 경우
            raise ValueError("Invalid move position")  # 오류 발생
        if move_position.x < 0 or move_position.y < 0 or move_position.x >= 9 or move_position.y >= 9:  # 이동할 위치가 범위를 벗어나는 경우
            return False  # False 반환
        if move_position not in character.get_abs_movable_positions():  # 이동할 수 없는 위치인 경우
            return False  # False 반환
        if self.board[4, int(move_position.x), int(move_position.y)] != 0:  # 이동할 위치에 캐릭터가 있는 경우
            return False  # False 반환
        diff = move_position - character.position
        return self.check_wall(character.position, diff)  # 이동좌표 사이에 벽이 있는지 확인

    def attack(self, character: Character):  # 공격
        if character.team == 1:  # 플레이어가 공격하는 경우
            nearest_enemy = None  # 가장 가까운 적
            nearest_distance = None  # 가장 가까운 거리
            for enemy in self.enemy_list:  # 적 리스트에 대해 반복
                if self.can_attack(character, enemy):  # 공격 가능한 적들 중 가장 가까운 적 선택
                    if nearest_enemy is None or character.position.distance(enemy.position) < nearest_distance:
                        nearest_enemy = enemy
                        nearest_distance = character.position.distance(enemy.position)
            if nearest_enemy is not None:  # 가장 가까운 적이 있는 경우
                return character.attack(nearest_enemy)  # 가장 가까운 적 공격
        elif character.team == 2:  # 적이 공격하는 경우
            if self.can_attack(character, self.player):  # 플레이어를 공격할 수 있는 경우
                return character.attack(self.player)  # 플레이어 공격

    def can_attack(self, character: Character, target: Character):  # 공격 가능한지 확인
        if character.team == target.team:  # 같은 팀인 경우
            return False  # False 반환
        if character.position not in target.get_abs_attackable_positions():  # 공격 가능한 위치가 아닌 경우
            return False  # False 반환
        return self.check_wall(character.position, target.position - character.position)  # 공격 가능한 위치 사이에 벽이 있는지 확인

    def next_turn(self):  # 다음 턴
        self.turn += 1  # 턴 증가
        for char in self.character_list:  # 캐릭터 리스트에 대해 반복
            char.update()  # 캐릭터 업데이트
        if self.turn % 2 == 1 and len(self.enemy_list) > 1:  # 홀수 턴인 경우
            self.enemy_circle_queue.forward()  # 적 순환 큐에서 다음 적으로 이동
        self.recalculate_board()  # 보드 재계산

    def auto_turn(self, character: Character = None, move_position: Vector2 = None):  # 자동 턴
        if character is None:  # 캐릭터가 주어지지 않은 경우
            character = self.current_character  # TODO: 현재 턴에 맞는 캐릭터 선택
        if move_position is not None:  # 이동할 위치가 주어진 경우
            if not self.move(move_position, character):  # 이동할 수 없는 경우
                raise Exception("Invalid move")
        self.attack(character)  # 공격
        self.next_turn()  # 다음 턴

    def get_movable_positions(self, character: Character = None):  # 이동 가능한 위치 반환
        if character is None:  # 캐릭터가 주어지지 않은 경우
            character = self.current_character  # 현재 캐릭터 선택
        movable_positions = [move for move in character.get_abs_movable_positions() if self.can_move(move, character)]  # 이동 가능한 위치 리스트
        if len(movable_positions) == 0:  # 이동 가능한 위치가 없는 경우
            return None  # None 반환
        return movable_positions  # 이동 가능한 위치 반환

    def check_wall(self, position: Vector2, vector: Vector2):  # 벽 검사: 벽이 있으면 False, 없으면 True
        if abs(vector.x) > abs(vector.y):  # x축 이동이 더 큰 경우
            if vector.x > 0:  # 우향
                return self.board[3, int(position.x), int(position.y)] == 0  # 우에 벽이 없는 경우 True 반환
            else:  # 좌향
                return self.board[2, int(position.x), int(position.y)] == 0  # 좌에 벽이 없는 경우 True 반환
        elif abs(vector.x) < abs(vector.y):  # y축 이동이 더 큰 경우
            if vector.y > 0:  # 상향
                return self.board[0, int(position.x), int(position.y)] == 0  # 상에 벽이 없는 경우 True 반환
            else:  # 하향
                return self.board[1, int(position.x), int(position.y)] == 0  # 하에 벽이 없는 경우 True 반환
        else:  # 대각선 이동인 경우
            return self.check_half_wall(position, vector)  # 반벽인지 확인

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

    def check_winner(self):  # 승자 확인
        if not self.player.is_active:  # 플레이어가 죽은 경우
            return 2  # 적 승리
        if not any([i.is_active for i in self.enemy_list]):  # 적이 모두 죽은 경우
            return 1  # 플레이어 승리
        return 0  # 아직 승자가 나오지 않은 경우

    def reward(self):  # 보상
        winner = self.check_winner()  # 승자 확인
        if winner == 1:  # 플레이어 승리
            return 1  # 1 반환
        elif winner == 2:  # 적 승리
            return -1  # -1 반환
        return 0  # 무승부인 경우 0 반환

    def print_board(self, index):
        print(self.board[index])
