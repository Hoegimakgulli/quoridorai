class EemeyFrame:
    def __init__(self, color, attackRange, moveRange, hp = 1):
        self.enemy_color = color
        self.enemy_attack_range = attackRange
        self.enemy_move_range = moveRange
        self.enemy_hp = hp

enemy_frames = []

# dataFrame (range)
diagonal_range = [[1,1], [-1,1], [-1,-1], [1,-1]]
cross_range    = [[1,0], [0,1], [0,-1], [-1,0]]

# dataSets (color)
red    = (255, 0, 0)
orange = (255, 100, 0)
yellow = (255, 255, 0)
green  = (0, 255, 0)
blue   = (0, 0, 255)
purple = (200, 0, 200)
grey   = (153, 153, 153)

# 해당 적 HP 넣어두기
def enemy_create_frame():
    enemy_frames.clear()
    enemy_frames.append(EemeyFrame(red, cross_range, cross_range))
    enemy_frames.append(EemeyFrame(orange, cross_range, cross_range))
    enemy_frames.append(EemeyFrame(yellow, cross_range, cross_range))
    enemy_frames.append(EemeyFrame(green, cross_range, cross_range))
    enemy_frames.append(EemeyFrame(blue, cross_range, cross_range))
    enemy_frames.append(EemeyFrame(purple, cross_range, cross_range))
    enemy_frames.append(EemeyFrame(grey, cross_range, cross_range))
    return enemy_frames

def EnemySelect(typeNum):
    if typeNum < len(enemy_frames):
        return enemy_frames[typeNum]
    else:
        print("해당 타입의 적은 존재하지않습니다.") 