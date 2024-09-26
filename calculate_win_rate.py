from ai.environment import Quoridor, Character
from ai.utils import Vector2
from ai.mcts import MCTS
from ai.enemy import path_finding
from ai.model import DQNAgent
from tqdm import tqdm


def calculate_win_rate(count):
    win = 0
    lose = 0
    brain = DQNAgent()
    brain.load_model(id=4999)
    if brain is None:
        raise ValueError("Model is not loaded")
    for _ in tqdm(range(count), desc="In Game: "):
        board = Quoridor()

        while board.check_winner() == 0:
            mcts = MCTS(board, brain, 0, 1)
            action = mcts.search().move if board.turn % 2 == 0 else path_finding(board.clone(), 9, 9)
            board.auto_turn(move_position=action)
            if board.check_winner() == 1:
                win += 1
                break
            elif board.check_winner() == 2:
                lose += 1
                break
    return win / count, lose / count


if __name__ == "__main__":
    win_rate, lose_rate = calculate_win_rate(100)
    print(f"Win rate: {win_rate * 100} %, Lose rate: {lose_rate* 100} %")
