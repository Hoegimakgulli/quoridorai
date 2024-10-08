from ai.environment import Quoridor, Character
from ai.utils import Vector2
from ai.mcts import MCTS
from ai.enemy import path_finding
from ai.model import DQNAgent
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
from copy import deepcopy


def move_position(scat: PathCollection, x, y):
    scat.set_offsets([x, y])
    return scat


if __name__ == "__main__":
    board = Quoridor()

    fig, ax = plt.subplots(figsize=(8, 8))

    player_point = ax.scatter(board.player.position.x, board.player.position.y, c="blue", s=1200)
    enemy_point_list = [(ax.scatter(enemy.position.x, enemy.position.y, c="red", s=1200), enemy) for enemy in board.enemy_list]

    board_list = [deepcopy(board.board[4])]

    ax.grid(True)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(9)])
    ax.set_yticks([i + 0.5 for i in range(9)])

    ax.tick_params(labelbottom=False, labelleft=False)
    brain = DQNAgent()
    brain.load_model(id=400)

    if brain is None:
        raise ValueError("Model is not loaded")

    def update(frame):
        print(f"Update: {frame}")
        global player_point, enemy_point_list
        mcts = MCTS(board, brain, 0, 1)
        if board.check_winner() != 0:
            return player_point, *enemy_point_list
        action = mcts.search().move if board.turn % 2 == 0 else path_finding(board.clone(), 9, 9)
        board.auto_turn(move_position=action)
        board_list.append(deepcopy(board.board[4]))
        player_point = move_position(player_point, board.player.position.x, board.player.position.y)
        if not board.player.is_active:
            player_point = move_position(player_point, -1, -1)
        for point, enemy in enemy_point_list:
            point = move_position(point, enemy.position.x, enemy.position.y)
            if not enemy.is_active:
                point = move_position(point, -1, -1)
        return player_point, *enemy_point_list

    def check_winner():
        frame = 0
        while board.check_winner() == 0 and frame < 100:
            yield frame
            print(f"Frame: {frame}")
            frame += 1
        yield frame

    ani = FuncAnimation(fig, update, frames=check_winner(), interval=500)

    ani.save("quoridor.gif", writer="pillow")

    open("history.txt", "w").write("\n".join([str(i) for i in board_list]))

    # plt.show()
