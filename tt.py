from ai.environment import Quoridor
import numpy as np

if __name__ == "__main__":
    quoridor = Quoridor()

    quoridor.print_board(4)
    poses = np.zeros((9, 9), dtype=int)
    dangerous_positions = quoridor.get_dangerous_positions()
    for pos in dangerous_positions:
        poses[pos.x, pos.y] = 1

    print(poses)
