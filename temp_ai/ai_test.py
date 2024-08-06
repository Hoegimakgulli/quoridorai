from collections import namedtuple
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from temp_env import Quoridor, Point, Character

### 데이터 전처리
row_dict = {}
for j in range(0, 50):
    row_dict[j] = j % 10
df = pd.read_csv("sample/quoridorai_map_example.csv").loc[:, (f"{i}" for i in range(9))]
df.drop([i * 10 + 9 for i in range(4)], axis=0, inplace=True)
df.rename(index=row_dict, inplace=True)
df_up, df_down, df_left, df_right, df_map = (df.iloc[i : i + 9, :] for i in range(0, 45, 9))
dfs = [df_up, df_down, df_right, df_left, df_map]

### Tuple 정의
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


### 메모리 생성
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


### 모델 생성
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(20 * 9 * 9, 500)
        self.fc2 = nn.Linear(500, 81)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 20 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


### Agent 생성
class DQNAgent:
    def __init__(self):
        self.model: DQN = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayMemory(10000)
        self.batch_size = 128
        self.gamma = 0.999

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state: torch.Tensor, available_actions: list[Point], epsilon=0.1):
        x = self.model(state)
        print(x)
        print(available_actions)

        values = [x[0, pos.x + pos.y * 9].item() if ((pos.x + pos.y * 9) >= 0 and (pos.x + pos.y * 9) < 81) else -1000 for pos in available_actions]
        print(values)
        max_value = max(values)
        max_index = values.index(max_value)
        print(max_index)

        if np.random.rand() > epsilon:
            return [*np.random.rand(len(available_actions))]
        else:
            return values

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + self.gamma * next_state_values
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pth")

    def load_model(self):
        self.model.load_state_dict(torch.load("model.pth"))


def train_agent(epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200):
    from mcts import MCTS

    agent = DQNAgent()
    env: Quoridor = Quoridor()
    for episode in range(1000):
        env.reset()
        state = env.get_board()
        done = False
        total_reward = 0

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * episode / epsilon_decay)
        while not done:
            mcts = MCTS(env, agent, epsilon)
            ## 플레이어 턴
            action = mcts.get_move()
            env.move(0, action, 1)
            env.attack(0, 1)
            reward = env.get_reward()

            next_state = env.get_board()
            agent.save_memory(state, action, next_state, reward)
            agent.update_model()

            total_reward += reward
            state = next_state

            if reward != 0:
                print(f"Episode {episode} finished with reward {total_reward}")
                done = True
                break

            ## 적 턴
            env.enemy_turn()
            if env.check_winner != 0:
                total_reward += env.get_reward()
                print(f"Episode {episode} finished with reward {total_reward}")
                done = True
                break

            state = env.get_board()

        agent.update_target_model()
        if episode % 10:
            agent.save_model()


if __name__ == "__main__":
    # agent = DQNAgent()
    # env: Quoridor = Quoridor()

    # state = env.get_board()

    # action = agent.get_action(state, env.get_legal_moves(1, 0))
    train_agent()
