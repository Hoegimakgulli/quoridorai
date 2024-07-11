class EemeyFrame:
    def __init__(self, color, attackRange, moveRange, hp = 1):
        self.enemy_color = color
        self.enemy_attackRange = attackRange
        self.enemy_moveRange = moveRange
        self.enemy_hp = hp

enemyFrames = []

# dataFrame (range)
diagonalRange = [[1,1], [-1,1], [-1,-1], [1,-1]]
crossRange    = [[1,0], [0,1], [0,-1], [-1,0]]

# dataSets (color)
red    = (255, 0, 0)
orange = (255, 100, 0)
yellow = (255, 255, 0)
green  = (0, 255, 0)
blue   = (0, 0, 255)
purple = (200, 0, 200)
grey   = (153, 153, 153)

# 해당 적 HP 넣어두기
def EnemyCreateFrame():
    enemyFrames.clear()
    enemyFrames.append(EemeyFrame(red, crossRange, crossRange))
    enemyFrames.append(EemeyFrame(orange, crossRange, crossRange))
    enemyFrames.append(EemeyFrame(yellow, crossRange, crossRange))
    enemyFrames.append(EemeyFrame(green, crossRange, crossRange))
    enemyFrames.append(EemeyFrame(blue, crossRange, crossRange))
    enemyFrames.append(EemeyFrame(purple, crossRange, crossRange))
    enemyFrames.append(EemeyFrame(grey, crossRange, crossRange))
    return enemyFrames

def EnemySelect(typeNum):
    if typeNum < len(enemyFrames):
        return enemyFrames[typeNum]
    else:
        print("해당 타입의 적은 존재하지않습니다.") 