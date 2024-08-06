import pygame
import random
import enemy_character_type
import numpy as np
import sys

# 재귀 한도 늘리기
sys.setrecursionlimit(10**7)


# 프로퍼티는 _"변수명"으로 설정
class EnemyData:
    def __init__(self, num, enemy_pos):
        self.enemy_num = num
        self._pos = enemy_pos
        self.enemy_color = None
        self.enemy_attack_range = None
        self.enemy_move_range = None
        self._hp = None

    def set_enemy_frame(self, color, attackRange, moveRange, enemy_hp):
        self.enemy_color = color
        self.enemy_attack_range = attackRange
        self.enemy_move_range = moveRange
        self._hp = enemy_hp

    @property
    def enemy_pos(self):
        return self._pos

    @enemy_pos.setter
    def enemy_pos(self, enemy_pos):
        if not 0 <= self._pos[0] < 9 and not 0 <= self._pos[1] < 9:
            raise ValueError("NUM : {} / 올바르지 않은 위치값입니다.".format(self.enemy_num))
        self._pos = enemy_pos

    @property
    def enemy_hp(self):
        return self._hp

    @enemy_hp.setter
    def enemy_hp(self, enemy_hp):
        if enemy._hp <= 0:
            print("NUM : {} / 기물이 처치되었습니다.".format(self.enemy_num))
        self._hp = enemy_hp


class Path:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y
        self.G = 0
        self.H = 0
        self.F = 0
        self.parent_node = None


# enemy 관련 변수
enemy_count = 2
enemy_datas = []
enemy_frames = enemy_character_type.enemy_create_frame()

# path_finding player, enemy 임시 저장 함수 실제로 움직일 때 사용하기 위해 넣어두는 함수 
# pygame안에서만 사용할것 AI에는 상관 X
player_tmp_box = None
enemy_tmp_box = None


# enemy 초기 스폰 위치 조정
def enemy_spawn():
    # 초기 인덱싱만 해준 데이터 생성
    for count in range(enemy_count):
        frame = enemy_frames[random.randint(0, 6)]
        enemy_datas.append(EnemyData(count, [0, 0]))
        enemy_datas[count].set_enemy_frame(frame.enemy_color, frame.enemy_attack_range, frame.enemy_move_range, frame.enemy_hp)

    # 위치 랜덤으로 조정해 pos 값 조정
    pos = [random.randint(0, 8), random.randint(0, 2)]
    for count in range(enemy_count):
        while enemy_find_with_pos(pos) is not False:
            pos = [random.randint(0, 8), random.randint(0, 2)]
        enemy_datas[count].enemy_pos = pos
    return 0


# pos 값으로 해당 인덱싱 넘버 뱉기
def enemy_find_with_pos(pos):
    for count in range(enemy_count):
        if enemy_datas[count].enemy_pos == pos:
            return enemy_datas[count].enemy_num
    return False


# enemy 그리기
def draw_enemy(screen):
    for count in range(len(enemy_datas)):
        drawPos = [enemy_datas[count].enemy_pos[0] * 50 + 25, enemy_datas[count].enemy_pos[1] * 50 + 25]
        pygame.draw.circle(screen, enemy_datas[count].enemy_color, drawPos, 25)


# A* 절차에 맞게 이동하는 절차 실행
def enemy_move(player, WIDTH, HEIGHT, wall_data):
    global player_tmp_box
    global enemy_tmp_box
    # 모든 적 기물에 대해 실행
    for count in range(len(enemy_datas)):
        player_tmp_box = player
        enemy_tmp_box = enemy_datas[count]
        path_finding(player.player_pos, enemy_datas[count].enemy_pos, WIDTH, HEIGHT, wall_data)


# A* 시작 player, enemy 매개변수로 받음
def path_finding(player_pos, enemy_pos, WIDTH, HEIGHT, wall_data):
    iter = 0
    size_x = WIDTH
    size_y = HEIGHT
    # path 리스트 초기화 [i][j]
    path_array = [[Path(i, j) for j in range(size_y)] for i in range(size_x)]

    start_node = path_array[enemy_pos[0]][enemy_pos[1]]
    target_node = path_array[player_pos[0]][player_pos[1]]

    open_list = [start_node]
    close_list = []
    final_path_list = []
    cur_node = None

    # 경로상 움직일 수 있는 좌표인지 확인 후 오픈리스트에 추가
    def open_list_add(x: int, y: int, move_dir: int):
        # check_can_move함수에 매개변수로 전달하기 위한 임시 변수
        if 0 <= x < size_x and 0 <= y < size_y and check_can_move([x, y]) and is_block_wall([cur_node.x, cur_node.y], move_dir, wall_data):
            node = path_array[x][y]
            if node not in close_list and node not in open_list:
                node.parent_node = cur_node
                node.G = cur_node.G + 1
                node.H = abs(node.x - target_node.x) + abs(node.y - target_node.y)
                node.F = node.G + node.H
                open_list.append(node)

    # 벽에 가로막혀 있는지 확인하는 함수
    def is_block_wall(start_pos, move_dir, wall_datas) -> bool:
        if not 0 <= move_dir <= 7:
            return ValueError("해당 방향은 올바르지않은 방향입니다. : wall_data (def is_block_wall)")

        pos_graph = start_pos[1] * 9 + start_pos[0]
        # 상하좌우 방향
        if move_dir % 2 == 0:
            if wall_datas[pos_graph][move_dir] == 0:
                return False
        # 나머지 대각방향 처리 1(↗), 3(↘), 5(↙), 7(↖)
        else:
            if wall_datas[pos_graph][(move_dir + 1) % 8] and wall_datas[pos_graph][(move_dir - 1) % 8]:
                return False

        return True

    while open_list:
        # 최대 iter 제한
        if iter > 1000:
            return False
        else:
            iter += 1

        cur_node = min(open_list, key=lambda node: (node.F, node.H))
        open_list.remove(cur_node)
        close_list.append(cur_node)

        if cur_node == target_node:
            print("A* end")
            targetcur_node = target_node.parent_node
            while targetcur_node != start_node:
                final_path_list.append(targetcur_node)
                targetcur_node = targetcur_node.parent_node
            final_path_list.append(start_node)
            final_path_list.reverse()
            change_enemy_pos(final_path_list)
            return final_path_list

        # 아직 대각쪽은 찾아보지 않음 상하좌우만 설정한 상태
        open_list_add(cur_node.x, cur_node.y + 1, 4)
        open_list_add(cur_node.x + 1, cur_node.y, 2)
        open_list_add(cur_node.x, cur_node.y - 1, 0)
        open_list_add(cur_node.x - 1, cur_node.y, 6)


def change_enemy_pos(path):
    enemy = enemy_tmp_box
    player = player_tmp_box
    if not check_can_attack(enemy, player):
        curMovePos = enemy.enemy_pos
        for pathCount in path:
            pathPos = [pathCount.x, pathCount.y]
            for movePos in enemy.enemy_move_range:
                trimPos = [curMovePos[0] + movePos[0], curMovePos[1] + movePos[1]]
                if trimPos == pathPos:
                    enemy.enemy_pos = trimPos
                    break
        # 움직이고 난 후 또 확인
        if check_can_attack(enemy, player):
            print("{} 색의 적이 공격".format(enemy.enemy_color))
            player.player_hp -= 1
    else:
        print("{} 색의 적이 공격".format(enemy.enemy_color))
        player.player_hp -= 1


# enemy가 player를 공격할 수 있는지 판독
def check_can_attack(enemy, player):
    for attackPos in enemy.enemy_attack_range:
        trimPos = [enemy.enemy_pos[0] + attackPos[0], enemy.enemy_pos[1] + attackPos[1]]
        if trimPos == player.player_pos:
            return True
    return False


# enemy가 해당 위치로 이동할 수 있는지 확인하는 함수
def check_can_move(movePos):
    for suvEnemy in enemy_datas:
        if movePos == suvEnemy.enemy_pos:
            return False
    return True
