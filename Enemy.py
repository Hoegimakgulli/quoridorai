import pygame
import random
import EnemyType
import numpy as np

class EnemyData:
    def __init__(self, num, pos, hp):
        self.enemy_num = num
        self.enemy_pos = pos
        self.enemy_hp = hp

    def set_enemyFrame(self, color, attackRange, moveRange):
        self.enemy_color = color
        self.enemy_attackRange = attackRange
        self.enemy_moveRange = moveRange
        
class Path:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y
    def Set_F(Self, _F):
        self.F = _F


pygame.init()

screen = pygame.display.set_mode([450, 450])
clock = pygame.time.Clock()
pygame.display.set_caption("쿼리도 슈팅 AI")

turn = 0
running = True

# player 임시 변수
playerPos = [4, 7]

# enemy 관련 변수
enemyCount = 2
enemyDatas = []
enemyFrames = EnemyType.EnemyCreateFrame()

def DrawBorad():
    rect_pos_size = [0, 0, 50, 50]
    screen.fill((255, 255, 255))
    # 홀수 짝수 순서로 검정색 하얀색으로 보드판 채워주기
    while rect_pos_size[0] != 450 or rect_pos_size[1] != 400:
        colorClsfc = rect_pos_size[0] + rect_pos_size[1]

        if ((colorClsfc//50) % 2) != 0:
            pygame.draw.rect(screen, (255, 255, 255), rect_pos_size)

        else:
            pygame.draw.rect(screen, (150, 150, 150), rect_pos_size)
        
        # 보드판 좌표 변경 부분
        rect_pos_size[0] += 50
        if rect_pos_size[0] == 500:
            rect_pos_size[0] = 0
            rect_pos_size[1] += 50

# def EnemyMove():
    

# enemy 초기 스폰 위치 조정
def EnemySpawn():
    # 초기 인덱싱만 해준 데이터 생성
    for count in range(enemyCount):
        frame = enemyFrames[random.randint(0,6)]
        enemyDatas.append(EnemyData(count, [0, 0], 10))
        enemyDatas[count].set_enemyFrame(frame.enemy_color, frame.enemy_attackRange, frame.enemy_moveRange)

    # 위치 랜덤으로 조정해 pos 값 조정
    pos = [random.randint(0, 8), random.randint(0, 2)]
    for count in range(enemyCount):
        while EnemyFindWithPos(pos) is not False:
            pos = [random.randint(0, 8), random.randint(0, 2)]
        enemyDatas[count].enemy_pos = pos
        
# pos 값으로 해당 인덱싱 넘버 뱉기
def EnemyFindWithPos(pos):
    for count in range(enemyCount):
        if enemyDatas[count].enemy_pos == pos:
            return enemyDatas[count].enemy_num
    return False

# enemy 그리기
def DrawEnemy():
    for count in range(len(enemyDatas)):
        drawPos = [enemyDatas[count].enemy_pos[0] * 50 + 25, enemyDatas[count].enemy_pos[1] * 50 + 25]
        pygame.draw.circle(screen, enemyDatas[count].enemy_color, drawPos, 25)

EnemySpawn()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                turn += 1

    DrawBorad()
    DrawEnemy()

    # player 그리기
    pygame.draw.circle(screen, (0, 125, 255), [playerPos[0] * 50 + 25, playerPos[1] * 50 + 25], 25)

    pygame.display.flip()

pygame.quit()