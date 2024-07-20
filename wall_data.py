import random
import pygame
# wall 데이터 삽입 일단 8 x 81 형태의 배열로 선언 i = 현재 위치 j = 8방향
# j - 0 2 4 6 (상우하좌) , 1 3 5 7 (↗, ↘, ↙, ↖)
wall_datas = [[0.1]*8 for _ in range(81)]

grey = (153, 153, 153)
green = (0, 200, 0)

def initialized():
    initl_wall()
    pass

def initl_wall():
    global wall_datas
    for wall_pos in range(81):
        x = wall_pos % 9
        y = wall_pos // 9
        
        if x == 0:
            wall_datas[wall_pos][6] = 0
        elif x == 8:
            wall_datas[wall_pos][2] = 0
        if y == 0:
            wall_datas[wall_pos][0] = 0
        elif y == 8:
            wall_datas[wall_pos][4] = 0

# 처음 맵에서 랜덤으로 벽 10개 생성하는 함수
def set_random_wall():
    initialized()
    global wall_data
    # 10번 반복
    for wall_count in range(10):
        # 0 ~ 80, 0 ~ 3
        x = random.randint(0, 7)
        y = random.randint(0, 7)
        rand_wall_pos = (y * 9) + x
        rand_wall_dir = random.randint(0, 3) * 2
        # 만약 랜덤한 위치에 벽이 이미 세워져있다면
        while wall_datas[rand_wall_pos][rand_wall_dir] == 0:
            x = random.randint(0, 7)
            y = random.randint(0, 7)
            rand_wall_pos = (y * 9) + x
            rand_wall_dir = random.randint(0, 3) * 2

        # 벽 데이터 변경 막힘 = 0
        # 아직 반벽 처리 안해줌!!!!!!!!!!!!!!!!!! 참고 바람
        if rand_wall_dir == 0 or rand_wall_dir == 4:
            wall_datas[rand_wall_pos][rand_wall_dir] = 0
            wall_datas[rand_wall_pos + 1][rand_wall_dir] = 0
            rand_wall_dir = 0 if rand_wall_dir == 4 else 4
            y += 1 if rand_wall_dir == 0 else -1
            rand_wall_pos = (y * 9) + x
            wall_datas[rand_wall_pos][rand_wall_dir] = 0
            wall_datas[rand_wall_pos + 1][rand_wall_dir] = 0
            pass

        elif rand_wall_dir == 2 or rand_wall_dir == 6:
            wall_datas[rand_wall_pos][rand_wall_dir] = 0
            wall_datas[rand_wall_pos + 9][rand_wall_dir] = 0
            rand_wall_dir = 2 if rand_wall_dir == 6 else 6
            x += 1 if rand_wall_dir == 6 else -1
            rand_wall_pos = (y * 9) + x
            wall_datas[rand_wall_pos][rand_wall_dir] = 0
            wall_datas[rand_wall_pos + 9][rand_wall_dir] = 0
            pass
    print("벽 생성 완료")

# 벽 그리기 함수
def draw_wall(screen):
    for draw_wall_pos in range(81):
        for draw_wall_dir in range(4):
            draw_wall_dir *= 2
            if wall_datas[draw_wall_pos][draw_wall_dir] == 0:
                x = draw_wall_pos % 9
                y = draw_wall_pos // 9
                # 벽이 위 아래에 있을 때
                if draw_wall_dir == 0 or draw_wall_dir == 4:
                    # 초기 시작 x, y 지정
                    x *= 50
                    y = (y + 1) * 50 if draw_wall_dir == 4 else y * 50
                    pygame.draw.line(screen, green, [x, y], [x + 50, y], 5)

                # 벽이 왼쪽 오른쪽에 있을 때 2 or 6
                elif draw_wall_dir == 2 or draw_wall_dir == 6:
                    x = (x + 1) * 50 if draw_wall_dir == 2 else x * 50
                    y *= 50
                    pygame.draw.line(screen, green, [x, y], [x, y + 50], 5)
    pass

# 현재 위치에서 계산하는 함수
def is_block_wall(pos, move_dir):
    pass