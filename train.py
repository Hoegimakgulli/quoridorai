from ai.model import DQNAgent
from ai.environment import Quoridor
from ai.mcts import MCTS
from ai.enemy import path_finding
from ai.utils import Vector2, CircularBuffer, mean
import torch
import numpy as np
from tqdm import tqdm


def train_agent(epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200):
    agent = DQNAgent()
    env: Quoridor = Quoridor()
    process_time: CircularBuffer = CircularBuffer(10)
    win_history: CircularBuffer = CircularBuffer(10)

    for episode in range(5001):
        env.reset()
        state = env.get_board()

        total_reward = 0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * episode / epsilon_decay)

        process_bar = tqdm(
            range(1000), desc=f"Episode {episode}: ", postfix=(None if episode == 0 else {"mean time": mean(process_time), "last time": process_time[-1], "win rate(last 10)": mean(win_history) * 100})
        )
        for turn in range(1000):
            mcts = MCTS(env, agent, epsilon, 100)

            if env.turn % 2 != 0:
                raise ValueError("플레이어의 턴이 아닙니다. : train_agent (def train_agent)")
            action = mcts.search().move  # 플레이어 턴이면 MCTS
            # print(action)
            env.auto_turn(move_position=action)  # 행동을 수행
            # env.print_board(4)  # 보드 출력
            reward = env.reward()  # 보상 계산

            next_state = env.get_board()  # 다음 상태

            total_reward += reward  # 총 보상 계산
            if type(action) != Vector2:
                continue
            if env.check_winner() == 0:  # 게임 종료 확인
                if env.turn % 2 != 1:
                    raise ValueError("적의 턴이 아닙니다. : train_agent (def train_agent)")
                enemy_action = path_finding(env.clone(), 9, 9)  # 아니면 path_finding(A*)
                env.auto_turn(move_position=enemy_action)
                reward = env.reward()
                next_state = env.get_board()
                total_reward += reward

            agent.save_memory(state, action.x + action.y * 9, next_state, torch.tensor([reward]))  # 메모리에 저장
            agent.update_model()  # 모델 업데이트

            state = next_state  # 상태 업데이트
            process_bar.update(1)
            if reward != 0:  # 게임 종료 확인
                # done = True
                break
        process_time.append(process_bar.format_dict["elapsed"])
        win_history.append(1 if reward == 1 else 0)

        process_bar.close()
        if reward == 1:  # 보상이 1이면 플레이어가 이긴 것
            print(f"Episode {episode} finished with reward {total_reward}, player won")  # 게임 종료 메시지 출력
        elif reward == -1:  # 보상이 -1이면 적이 이긴 것
            print(f"Episode {episode} finished with reward {total_reward}, enemy won")  # 게임 종료 메시지 출력

        agent.update_target_model()
        if episode % 100 == 0:
            agent.save_model(id=episode)


if __name__ == "__main__":
    train_agent()
