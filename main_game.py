import enemy_character
import player_character
import wall_data
import numpy as np
import pygame

pygame.init()

SCALE = 50
WIDTH = 9
HEIGHT = 9
screen = pygame.display.set_mode([WIDTH * SCALE, HEIGHT * SCALE])
clock = pygame.time.Clock()
pygame.display.set_caption("쿼리도 슈팅 AI")

turn = 1
running = True
turn_anchor = 1


def initialized():
    global turn_anchor, turn
    turn_anchor = 1
    turn += 1


# convert data main_game -> temp_env
# 필요한 데이터 적, 아군, 벽 return 값으로 전달 [9][9][5]
def convert_pygame_data_set():
    # 새로운 데이터 배열을 만드는거라 기존에 temp_env에 있는 데이터 셋이랑 합치는 작업이 필요할 것으로 예상
    env_data = np.zeros((9, 9, 5), dtype=int)

    # 플레이어 데이터 저장
    for player in player_character.player_datas:
        env_data[player.player_pos[0], player.player_pos[1], 4] = 1

    # 적 데이터 저장
    for enemy in enemy_character.enemy_datas:
        env_data[enemy.enemy_pos[0], enemy.enemy_pos[1], 4] = 3

    # 벽 데이터 저장 : 해당 지점에서 이동할 때 그 방향이 막혀있는 경우 가중치 0으로 설정
    for wall in range(81):
        wall_posX = wall % 9
        wall_posY = wall // 9
        temp_move_dir = 0
        for move_dir in [0, 4, 6, 2]:  # 각 상(↑), 우(→), 하(↓), 좌(←) 순서 : temp_env 상하좌우 순
            if wall_data.wall_datas[wall][move_dir] == 0:  # 벽에 막힘을 기준하는 값 (미정)
                env_data[wall_posX, wall_posY, temp_move_dir] = 1  # env_data에도 적용
            temp_move_dir += 1

    return env_data


# convert data temp_env -> main_game
# 매개변수 env_data로 받기
def convert_env_data_set(env_data):
    # 벽 데이터 변환
    for wall in range(81):
        wall_posX = wall % 9
        wall_posY = wall // 9
        pygame_move_dir = 0
        for move_dir in [0, 3, 2, 1]:
            if env_data[wall_posX, wall_posY, move_dir] == 0:  # 벽에 막힘 처리
                wall_data.wall_datas[wall][pygame_move_dir * 2] == 0
            pygame_move_dir += 1
    pass


def DrawBorad():
    rect_pos_size = [0, 0, 50, 50]
    screen.fill((255, 255, 255))
    # 홀수 짝수 순서로 검정색 하얀색으로 보드판 채워주기
    while rect_pos_size[0] != 450 or rect_pos_size[1] != 400:
        colorClsfc = rect_pos_size[0] + rect_pos_size[1]

        if ((colorClsfc // 50) % 2) != 0:
            pygame.draw.rect(screen, (255, 255, 255), rect_pos_size)

        else:
            pygame.draw.rect(screen, (30, 30, 30), rect_pos_size)

        # 보드판 좌표 변경 부분
        rect_pos_size[0] += 50
        if rect_pos_size[0] == 500:
            rect_pos_size[0] = 0
            rect_pos_size[1] += 50


# GameRule 모음
def enemy_win():
    if len(enemy_character.EnemyData) <= 0:
        print("enemy_win")


def player_win():
    if len(enemy_character.EnemyData) <= 0:
        print("player_win")


# 시작 메인 함수 (게임 룰 관리 및 유닛 이동 절차)
enemy_character.enemy_spawn()
player_character.player_spawn()
# wall_data.py안에 player 데이터와 enemy 데이터 각각 전달
wall_data.set_random_wall(player_character.player_datas, enemy_character.enemy_datas)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                initialized()
        # test key setting KEYDOWN으로 묶어둠
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                temp_env_data = convert_pygame_data_set()
                print(temp_env_data)

    # enemyturn 실행
    if turn % 2 == 0 and turn_anchor == 1:
        print("Start EnemyTurn")
        turn_anchor = 0
        if len(player_character.player_datas) > 0:
            enemy_character.enemy_move(player_character.player_datas[0], WIDTH, HEIGHT, wall_data.wall_datas)

    # playerturn 실행
    if turn % 2 == 1 and turn_anchor == 1:
        print("Start playerTurn")
        turn_anchor = 0
        # player 움직이는 함수 실행

    DrawBorad()
    wall_data.draw_wall(screen)
    enemy_character.draw_enemy(screen)
    player_character.draw_player(screen)

    pygame.display.flip()

pygame.quit()
