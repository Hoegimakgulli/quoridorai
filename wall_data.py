import random
import pygame

# wall 데이터 삽입 일단 8 x 81 형태의 배열로 선언 i = 현재 위치 j = 8방향
# j - 0 2 4 6 (상우하좌) , 1 3 5 7 (↗, ↘, ↙, ↖)
wall_datas = [[0.1] * 8 for _ in range(81)]

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
    global wall_datas

    # 벽으로 인해 가둬지는 기물이 있는지 확인하는 함수
    def is_confine_wall(tmp_wall_data):
        check_move_point = [[1] * 4 for _ in range(81)]
        check_list = [0]

        while check_list:
            select_pos = check_list[0]
            del check_list[0]

            # 4방향 전부 검사 후 벽으로 안막혀 있으면 리스트에 추가
            for check_dir in [0, 2, 4, 6]:
                if not tmp_wall_data[select_pos][check_dir] == 0:
                    if check_dir == 0:
                        check_pos = select_pos - 9
                    elif check_dir == 2:
                        check_pos = select_pos + 1
                    elif check_dir == 4:
                        check_pos = select_pos + 9
                    else:
                        check_pos = select_pos - 1

                    if check_move_point[check_pos][check_dir // 2] == 1:
                        check_list.append(check_pos)
                        check_move_point[check_pos][check_dir // 2] = 0
                # 벽으로 막힌경우
                else:
                    check_move_point[check_pos][check_dir // 2] = 0
        # 만약 가보지 못한 곳이 존재한다면 막혀있는 공간이 있다고 판단
        for check_count in range(81):
            for check_dir in range(4):
                if check_move_point[check_count][check_dir] == 1:
                    return True

        return False

    def set_wall_data(pos, dir, wall, posx, posy):
        if dir == 0 or dir == 4:
            wall[pos][dir] = 0
            wall[pos + 1][dir] = 0
            dir = 0 if dir == 4 else 4
            posy += 1 if dir == 0 else -1
            pos = (posy * 9) + posx
            wall[pos][dir] = 0
            wall[pos + 1][dir] = 0

        elif dir == 2 or dir == 6:
            wall[pos][dir] = 0
            wall[pos + 9][dir] = 0
            dir = 2 if dir == 6 else 6
            posx += 1 if dir == 6 else -1
            pos = (posy * 9) + posx
            wall[pos][dir] = 0
            wall[pos + 9][dir] = 0

        return wall

    # 10번 반복
    for _ in range(10):
        # 0 ~ 80, 0 ~ 3
        x = random.randint(0, 7)
        y = random.randint(0, 7)
        rand_wall_pos = (y * 9) + x
        rand_wall_dir = random.randint(0, 3) * 2
        # 만약 랜덤한 위치에 벽이 이미 세워져있다면
        # or 벽이 기존에 있던 벽을 관통한다면

        # 벽 관통 여부 초기화 함수
        penetrate_x = x + 1 if rand_wall_dir == 2 else x
        penetrate_y = y + 1 if rand_wall_dir == 4 else y
        build_dir = 2 if rand_wall_dir == 0 or rand_wall_dir == 4 else 4

        # tmp_wall_datas = set_wall_data(rand_wall_pos, rand_wall_dir, wall_datas.copy(), x, y)

        # 벽 설치 가능 여부 확인 아래 조건에 걸리면 다시 랜덤으로 뽑아서 쓰기
        while wall_datas[rand_wall_pos][rand_wall_dir] == 0 or is_penetrate_wall(penetrate_x, penetrate_y, build_dir):  # or is_confine_wall(tmp_wall_datas):
            x = random.randint(0, 7)
            y = random.randint(0, 7)
            rand_wall_pos = (y * 9) + x
            rand_wall_dir = random.randint(0, 3) * 2

            penetrate_x = x + 1 if rand_wall_dir == 2 else x
            penetrate_y = y + 1 if rand_wall_dir == 4 else y
            build_dir = 2 if rand_wall_dir == 0 or rand_wall_dir == 4 else 4
            # tmp_wall_datas = set_wall_data(rand_wall_pos, rand_wall_dir, wall_datas.copy(), x, y)

        wall_datas = set_wall_data(rand_wall_pos, rand_wall_dir, wall_datas.copy(), x, y)
        # 벽 데이터 변경 막힘 = 0
        # if rand_wall_dir == 0 or rand_wall_dir == 4:
        #     wall_datas[rand_wall_pos][rand_wall_dir] = 0
        #     wall_datas[rand_wall_pos + 1][rand_wall_dir] = 0
        #     rand_wall_dir = 0 if rand_wall_dir == 4 else 4
        #     y += 1 if rand_wall_dir == 0 else -1
        #     rand_wall_pos = (y * 9) + x
        #     wall_datas[rand_wall_pos][rand_wall_dir] = 0
        #     wall_datas[rand_wall_pos + 1][rand_wall_dir] = 0
        #     pass

        # elif rand_wall_dir == 2 or rand_wall_dir == 6:
        #     wall_datas[rand_wall_pos][rand_wall_dir] = 0
        #     wall_datas[rand_wall_pos + 9][rand_wall_dir] = 0
        #     rand_wall_dir = 2 if rand_wall_dir == 6 else 6
        #     x += 1 if rand_wall_dir == 6 else -1
        #     rand_wall_pos = (y * 9) + x
        #     wall_datas[rand_wall_pos][rand_wall_dir] = 0
        #     wall_datas[rand_wall_pos + 9][rand_wall_dir] = 0
        #     pass
    print("벽 생성 완료")


# 벽끼리 관통되어 생성될 수 있는지 확인하는 함수 관통될 시 True 반환
def is_penetrate_wall(start_posx: int, start_posy: int, build_dir: int) -> bool:
    global wall_datas
    trim_posx = start_posx
    trim_posy = start_posy

    if build_dir == 0:  # 위쪽
        trim_posx += -1
        trim_posy += -1
        if wall_datas[trim_posy * 9 + trim_posx][build_dir] == 0 and wall_datas[trim_posy * 9 + trim_posx + 1][build_dir] == 0:
            return True
        pass
    elif build_dir == 2:  # 오른쪽
        if wall_datas[trim_posy * 9 + trim_posx][build_dir] == 0 and wall_datas[(trim_posy - 1) * 9 + trim_posx][build_dir] == 0:
            return True
        pass
    elif build_dir == 4:  # 아래쪽
        if wall_datas[trim_posy * 9 + trim_posx][build_dir] == 0 and wall_datas[trim_posy * 9 + trim_posx - 1][build_dir] == 0:
            return True
        pass
    elif build_dir == 6:  # 왼쪽
        trim_posx += -1
        trim_posy += -1
        if wall_datas[trim_posy * 9 + trim_posx][build_dir] == 0 and wall_datas[(trim_posy + 1) * 9 + trim_posx][build_dir] == 0:
            return True
        pass
    else:
        raise ValueError("선택하신 방향은 올바르지않은 방향입니다.")
    return False


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
# 움직이기전 좌표, 움직이고난 좌표, 미리 벽처리한 벽 데이터
# enemy_character & player_character에서 사용할 예정 추후 이식하고 삭제할 예정
def is_block_wall(start_pos, move_dir, wall_datas) -> bool:
    if not 0 <= move_dir <= 7:
        return ValueError("해당 방향은 올바르지않은 방향입니다. : wall_data (def is_block_wall)")

    pos_graph = start_pos[1] * 9 + start_pos[0]
    # 상하좌우 방향
    if move_dir % 2 == 0:
        if wall_datas[pos_graph][move_dir] == 0:
            return False
    # 나머지 대각방향 처리 1(↗), 3(↘), 5(↙), 7(↖)
    else:
        if wall_datas[pos_graph][(move_dir + 1) % 8] and wall_datas[pos_graph][(move_dir - 1) % 8]:
            return False

    return True