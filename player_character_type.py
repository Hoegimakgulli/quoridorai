class PlayerFrame:
    def __init__(self, color, attackRange, moveRange, hp = 1):
        self.player_color = color
        self.player_attack_range = attackRange
        self.player_move_range = moveRange
        self.player_hp = hp

player_frames = []

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
def player_create_frame():
    player_frames.clear()
    player_frames.append(PlayerFrame(red, cross_range, cross_range))
    player_frames.append(PlayerFrame(orange, cross_range, cross_range))
    player_frames.append(PlayerFrame(yellow, cross_range, cross_range))
    player_frames.append(PlayerFrame(green, cross_range, cross_range))
    player_frames.append(PlayerFrame(blue, cross_range, cross_range))
    player_frames.append(PlayerFrame(purple, cross_range, cross_range))
    player_frames.append(PlayerFrame(grey, cross_range, cross_range))
    return player_frames

def PlayerSelect(typeNum):
    if typeNum < len(player_frames):
        return player_frames[typeNum]
    else:
        print("해당 타입의 플레이어는 존재하지않습니다.") 