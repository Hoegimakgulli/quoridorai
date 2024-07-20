import enemy_character
import player_character
import wall_data
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

def DrawBorad():
    rect_pos_size = [0, 0, 50, 50]
    screen.fill((255, 255, 255))
    # 홀수 짝수 순서로 검정색 하얀색으로 보드판 채워주기
    while rect_pos_size[0] != 450 or rect_pos_size[1] != 400:
        colorClsfc = rect_pos_size[0] + rect_pos_size[1]

        if ((colorClsfc//50) % 2) != 0:
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
wall_data.set_random_wall()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                initialized()

    # enemyturn 실행
    if turn % 2 == 0 and turn_anchor == 1:
        print("Start EnemyTurn")
        turn_anchor = 0
        if len(player_character.player_datas) > 0:
            enemy_character.enemy_move(player_character.player_datas[0], WIDTH, HEIGHT)

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