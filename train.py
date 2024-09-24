from ai.model import DQNAgent
from ai.environment import Quoridor
from ai.mcts import MCTS
import torch
import numpy as np


def train_agent(epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200):
    agent = DQNAgent()
    env: Quoridor = Quoridor()
    for episode in range(1000):
        env.reset()
        state = env.get_board()

        total_reward = 0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * episode / epsilon_decay)
        for turn in range(1000):
            mcts = MCTS(env, agent, epsilon)
            ## 플레이어 턴
            action = mcts.search().move
            print(action)
            print(env.move(0, action, 1))
            env.attack(0, 1, show_log=True)
            env.print_board(4)
            reward = env.get_reward()

            next_state = env.get_board()
            agent.save_memory(state, action.x + action.y * 9, next_state, torch.tensor([reward]))
            agent.update_model()

            total_reward += reward
            state = next_state

            if reward != 0:
                print(f"Episode {episode} finished with reward {total_reward}, player won")
                # done = True
                break

            ## 적 턴
            print(env.enemy_turn())
            print(env.print_board(4))
            if env.check_winner() != 0:
                total_reward += env.get_reward()
                print(f"Episode {episode} finished with reward {total_reward}, enemy won")
                # done = True
                break

            state = env.get_board()

        agent.update_target_model()
        if episode % 10:
            agent.save_model()
