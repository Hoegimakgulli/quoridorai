import Enemy
import Player
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
turnAnchor = 1

def Initialized():
    turnAnchor = 1

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
def EnemyWin():
    if len(Enemy.EnemyData) <= 0:
        print("EnemyWin")

def PlayerWin():
    if len(Enemy.EnemyData) <= 0:
        print("PlayerWin")

# 시작 메인 함수 (게임 룰 관리 및 유닛 이동 절차)
Enemy.EnemySpawn()
Player.PlayerSpawn()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                turn += 1
                turnAnchor = 1

    # enemyturn 실행
    if turn % 2 == 0 and turnAnchor == 1:
        print("Start EnemyTurn")
        turnAnchor = 0
        Enemy.EnemyMove(Player.playerDatas[0], WIDTH, HEIGHT)

    # playerturn 실행
    if turn % 2 == 1 and turnAnchor == 1:
        print("Start playerTurn")
        turnAnchor = 0
        # player 움직이는 함수 실행

    DrawBorad()
    Enemy.DrawEnemy(screen)
    Player.DrawPlayer(screen)

    pygame.display.flip()

pygame.quit()