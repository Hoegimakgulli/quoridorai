import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ai.environment import Quoridor, Character, Wall
from ai.utils import Vector2
from collections import namedtuple

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
    def __init__(self, device):
        self.device = device
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=1).to(self.device)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1).to(self.device)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1).to(self.device)
        self.fc1 = nn.Linear(20 * 9 * 9, 500).to(self.device)
        self.fc2 = nn.Linear(500, 81).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x)).to(self.device)
        x = F.relu(self.conv2(x)).to(self.device)
        x = F.relu(self.conv3(x)).to(self.device)
        x = x.view(-1, 20 * 9 * 9).to(self.device)
        x = F.relu(self.fc1(x)).to(self.device)
        x = self.fc2(x)
        return x


### Agent 생성
class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model: DQN = DQN(self.device)
        self.target_model = DQN(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayMemory(10000)
        self.batch_size = 128
        self.gamma = 0.999

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state: torch.Tensor, available_actions: list[Vector2], epsilon=0.1):
        x = self.model(state)
        # print(x)
        # print(available_actions)
        if available_actions == None or len(available_actions) == 0:
            return []
        values = [x[0, pos.x + pos.y * 9].item() if ((pos.x + pos.y * 9) >= 0 and (pos.x + pos.y * 9) < 81) else -1000 for pos in available_actions]
        # print(values)
        max_value = max(values)
        max_index = values.index(max_value)
        # print(max_index)

        if np.random.rand() > epsilon:
            random_weight = [*np.random.rand(len(available_actions))]
            value_list = [(action, random_weight[i]) for i, action in enumerate(available_actions)]
            return value_list

        else:
            value_list = [(action, values[i]) for i, action in enumerate(available_actions)]
            return value_list

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        # action_batch = torch.cat(batch.action)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        # print(action_batch)
        reward_batch = torch.cat(batch.reward).to(self.device)

        state_action_values = self.model(state_batch).gather(1, action_batch).to(self.device)
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach().to(self.device)
        expected_state_action_values = reward_batch + self.gamma * next_state_values
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pth")

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load("model.pth", map_location=self.device, weights_only=True))
            # print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model file not found. Please check the path.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise ValueError("Model is not loaded")
