import pygame
import random
import EnemyType
import numpy as np
import sys
# 재귀 한도 늘리기
sys.setrecursionlimit(10**7)

class EnemyData:
    def __init__(self, num, pos):
        self.enemy_num = num
        self.enemy_pos = pos

    def SetEnemyFrame(self, color, attackRange, moveRange, hp):
        self.enemy_color = color
        self.enemy_attackRange = attackRange
        self.enemy_moveRange = moveRange
        self.enemy_hp = hp
        
class Path:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y
        self.G = 0
        self.H = 0
        self.F = 0
        self.parentNode = None

# enemy 관련 변수
enemyCount = 2
enemyDatas = []
enemyFrames = EnemyType.EnemyCreateFrame()

# enemy 초기 스폰 위치 조정
def EnemySpawn():
    # 초기 인덱싱만 해준 데이터 생성
    for count in range(enemyCount):
        frame = enemyFrames[random.randint(0,6)]
        enemyDatas.append(EnemyData(count, [0, 0]))
        enemyDatas[count].SetEnemyFrame(frame.enemy_color, \
            frame.enemy_attackRange, frame.enemy_moveRange, frame.enemy_hp)

    # 위치 랜덤으로 조정해 pos 값 조정
    pos = [random.randint(0, 8), random.randint(0, 2)]
    for count in range(enemyCount):
        while EnemyFindWithPos(pos) is not False:
            pos = [random.randint(0, 8), random.randint(0, 2)]
        enemyDatas[count].enemy_pos = pos
    return 0
        
# pos 값으로 해당 인덱싱 넘버 뱉기
def EnemyFindWithPos(pos):
    for count in range(enemyCount):
        if enemyDatas[count].enemy_pos == pos:
            return enemyDatas[count].enemy_num
    return False

# enemy 그리기
def DrawEnemy(screen):
    for count in range(len(enemyDatas)):
        drawPos = [enemyDatas[count].enemy_pos[0] * 50 + 25, enemyDatas[count].enemy_pos[1] * 50 + 25]
        pygame.draw.circle(screen, enemyDatas[count].enemy_color, drawPos, 25)

# A* 절차에 맞게 이동하는 절차 실행
def EnemyMove(player, WIDTH, HEIGHT):
    # 모든 적 기물에 대해 실행
    for count in range(len(enemyDatas)):
        PathFinding(player, enemyDatas[count], WIDTH, HEIGHT)

# A* 시작 player, enemy 매개변수로 받음
def PathFinding(player, enemy, WIDTH, HEIGHT):
    sizeX = WIDTH
    sizeY = HEIGHT
    # path 리스트 초기화 [i][j]       
    pathArray = [[Path(i, j) for j in range(sizeY)] for i in range(sizeX)]

    startNode = pathArray[enemy.enemy_pos[0]][enemy.enemy_pos[1]]
    targetNode = pathArray[player.player_pos[0]][player.player_pos[0]]

    openList = [startNode]
    closeList = []
    finalPathList = []
    curNode = None

    def openListAdd(x, y):
        print("openListSet")
        if 0 <= x < sizeX and 0 <= y < sizeY:
            node = pathArray[x][y]
            if node not in closeList and node not in openList:
                node.parentNode = curNode
                node.G = curNode.G + 1
                node.H = abs(node.x - targetNode.x) + abs(node.y - targetNode.y)
                node.F = node.G + node.H
                openList.append(node)

    while openList:
        curNode = min(openList, key=lambda node: (node.F, node.H))
        openList.remove(curNode)
        closeList.append(curNode)

        if curNode == targetNode:
            print("A* end")
            targetCurNode = targetNode.parentNode
            while targetCurNode != startNode:
                finalPathList.append(targetCurNode)
                targetCurNode = targetCurNode.parentNode
            finalPathList.append(startNode)
            finalPathList.reverse()
            ChangeEnemyPos(finalPathList, enemy, player)
            return 0

        openListAdd(curNode.x, curNode.y + 1)
        openListAdd(curNode.x + 1, curNode.y)
        openListAdd(curNode.x, curNode.y - 1)
        openListAdd(curNode.x - 1, curNode.y)
        
        if not finalPathList:
            return PathFinding(player, enemy, sizeX, sizeY)

def ChangeEnemyPos(path, enemy, player):
    if not CheckCanAttack(enemy, player):
        curMovePos = enemy.enemy_pos
        for pathCount in path:
            pathPos = [pathCount.x, pathCount.y]
            for movePos in enemy.enemy_moveRange:
                trimPos = curMovePos + movePos
                if trimPos == pathPos:
                    enemy.enemy_pos = trimPos
                    break
        if CheckCanAttack(enemy, player):
            player_hp -= 1

# enemy가 player를 공격할 수 있는지 판독
def CheckCanAttack(enemy, player):
    for attackPos in enemy.enemy_attackRange:
        trimPos = enemy.enemy_pos + attackPos
        if trimPos == player.player_pos:
            return True
    return False