from ai.environment import Quoridor, Character
from ai.utils import Vector2
from ai.mcts import MCTS
from ai.enemy import path_finding
from ai.model import DQNAgent
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection


def move_position(scat: PathCollection, x, y):
    scat.set_offsets([x, y])
    return scat


if __name__ == "__main__":
    board = Quoridor()
    board.player = board.player.clone()
    board.enemy_list = [board.enemy_list[0]]
    board.turn = board.turn
    board.recalculate_board()

    fig, ax = plt.subplots(figsize=(8, 8))

    player_point = ax.scatter(board.player.position.x, board.player.position.y, c="blue", s=1200)
    enemy_point = ax.scatter(board.enemy_list[0].position.x, board.enemy_list[0].position.y, c="red", s=1200)

    ax.grid(True)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(9)])
    ax.set_yticks([i + 0.5 for i in range(9)])

    ax.tick_params(labelbottom=False, labelleft=False)
    brain = DQNAgent()
    brain.load_model()

    if brain is None:
        raise ValueError("Model is not loaded")

    def update(frame):
        global player_point, enemy_point
        mcts = MCTS(board, brain, 0, 1)
        action = mcts.search().move if board.turn % 2 == 0 else path_finding(board.clone(), 9, 9)
        board.auto_turn(move_position=action)
        player_point = move_position(player_point, board.player.position.x, board.player.position.y)
        enemy_point = move_position(enemy_point, board.enemy_list[0].position.x, board.enemy_list[0].position.y)
        return player_point, enemy_point

    def check_winner():
        frame = 0
        while board.check_winner() == 0:
            yield frame
            frame += 1
        yield frame

    ani = FuncAnimation(fig, update, frames=check_winner(), interval=500)

    ani.save("quoridor.gif", writer="pillow")

    # plt.show()
