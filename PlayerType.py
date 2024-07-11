class PlayerFrame:
    def __init__(self, color, attackRange, moveRange, hp = 1):
        self.player_color = color
        self.player_attackRange = attackRange
        self.player_moveRange = moveRange
        self.player_hp = hp

playerFrames = []

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
def PlayerCreateFrame():
    playerFrames.clear()
    playerFrames.append(PlayerFrame(red, crossRange, crossRange))
    playerFrames.append(PlayerFrame(orange, crossRange, crossRange))
    playerFrames.append(PlayerFrame(yellow, crossRange, crossRange))
    playerFrames.append(PlayerFrame(green, crossRange, crossRange))
    playerFrames.append(PlayerFrame(blue, crossRange, crossRange))
    playerFrames.append(PlayerFrame(purple, crossRange, crossRange))
    playerFrames.append(PlayerFrame(grey, crossRange, crossRange))
    return playerFrames

def PlayerSelect(typeNum):
    if typeNum < len(playerFrames):
        return playerFrames[typeNum]
    else:
        print("해당 타입의 플레이어는 존재하지않습니다.") 