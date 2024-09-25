from ai.environment import Quoridor, Character
from ai.utils import Vector2


class Path:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y
        self.G = 0
        self.H = 0
        self.F = 0
        self.parent_node = None


# A* 시작 state 매개변수로 받음 (state: Quoridor)
def path_finding(state: Quoridor, WIDTH, HEIGHT):
    # global target_player
    target_player: Character = state.player
    current_enemy: Character = state.current_character
    if state.turn % 2 == 0:
        raise ValueError("적의 턴이 아닙니다. : path_finding (def path_finding)")
    iter = 0
    size_x = WIDTH
    size_y = HEIGHT
    # path 리스트 초기화 [i][j]
    path_array = [[Path(i, j) for j in range(size_y)] for i in range(size_x)]
    # print(player_pos)
    start_node: Path = path_array[current_enemy.position.x][current_enemy.position.y]
    target_node: Path = path_array[target_player.position.x][target_player.position.y]

    open_list = [start_node]
    close_list = []
    final_path_list = []
    path_list_pos = []
    cur_node = None

    # 경로상 움직일 수 있는 좌표인지 확인 후 오픈리스트에 추가
    def open_list_add(x: int, y: int):
        # check_can_move함수에 매개변수로 전달하기 위한 임시 변수
        if 0 <= x < size_x and 0 <= y < size_y and state.check_wall(Vector2(cur_node.x, cur_node.y), Vector2(x - cur_node.x, y - cur_node.y)) and state.board[4, x, y] != 2:
            node = path_array[x][y]
            if node not in close_list and node not in open_list:
                node.parent_node = cur_node
                node.G = cur_node.G + 1
                node.H = abs(node.x - target_node.x) + abs(node.y - target_node.y)
                node.F = node.G + node.H
                open_list.append(node)

    while open_list:
        # 최대 iter 제한
        if iter > 1000:
            return False
        else:
            iter += 1

        cur_node = min(open_list, key=lambda node: (node.F, node.H))
        open_list.remove(cur_node)
        close_list.append(cur_node)

        if cur_node == target_node:
            # print("A* end")
            targetcur_node: Path = target_node.parent_node
            while targetcur_node != start_node:
                final_path_list.append(targetcur_node)
                targetcur_node = targetcur_node.parent_node
            final_path_list.append(start_node)
            final_path_list.reverse()

            for path in final_path_list:
                path_list_pos.append(Vector2(path.x, path.y))

        # 아직 대각쪽은 찾아보지 않음 상하좌우만 설정한 상태
        open_list_add(cur_node.x, cur_node.y + 1)
        open_list_add(cur_node.x + 1, cur_node.y)
        open_list_add(cur_node.x, cur_node.y - 1)
        open_list_add(cur_node.x - 1, cur_node.y)

    # print(path_list_pos)
    if len(path_list_pos) < 2:
        return None
    return Vector2(list=path_list_pos[1])
