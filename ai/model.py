import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ai.environment import Quoridor, Character, Wall
from ai.utils import Vector2
from collections import namedtuple
import os
import re

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

        self.model: DQN = DQN(self.device)  # 모델 생성
        self.target_model = DQN(self.device)  # 타겟 모델 생성
        self.target_model.load_state_dict(self.model.state_dict())  # 타겟 모델에 모델 복사
        self.target_model.eval()  # 타겟 모델 평가모드로 설정
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # 옵티마이저 생성
        self.memory = ReplayMemory(10000)  # 메모리 생성
        self.batch_size = 128  # 배치 사이즈
        self.gamma = 0.999  # 감마 값

    def update_target_model(self):  # 타겟 모델 업데이트
        self.target_model.load_state_dict(self.model.state_dict())  # 모델 복사

    def get_action(self, state: torch.Tensor, available_actions: list[Vector2], epsilon=0.1):  # 액션 선택
        x = self.model(state)  # 모델에 상태 입력
        # print(x)
        # print(available_actions)
        if available_actions == None or len(available_actions) == 0:  # 이동 가능한 액션이 없을 때
            return []  # 빈 리스트 반환
        values = [x[0, pos.x + pos.y * 9].item() if ((pos.x + pos.y * 9) >= 0 and (pos.x + pos.y * 9) < 81) else -1000 for pos in available_actions]  # 이동 가능한 액션에 대한 값 계산
        # print(values)
        max_value = max(values)  # 최대값 계산
        max_index = values.index(max_value)  # 최대값 인덱스 계산
        # print(max_index)

        if np.random.rand() > epsilon:  # 랜덤으로 액션 선택
            random_weight = [*np.random.rand(len(available_actions))]  # 랜덤 가중치 생성
            value_list = [(action, random_weight[i]) for i, action in enumerate(available_actions)]  # 가중치 리스트 생성
            return value_list  # 가중치 리스트 반환

        else:  # 최대값으로 액션 선택
            value_list = [(action, values[i]) for i, action in enumerate(available_actions)]  # 가중치 리스트 생성
            return value_list  # 가중치 리스트 반환

    def update_model(self):  # 모델 업데이트
        if len(self.memory) < self.batch_size:  # 메모리가 배치 사이즈보다 작을 때
            return  # 종료

        transitions = self.memory.sample(self.batch_size)  # 메모리에서 샘플링
        batch = Transition(*zip(*transitions))  # 튜플로 변환

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.device)  # 다음 상태가 존재하는지 확인
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)  # 다음 상태 존재 시 다음 상태를 캣으로 결합

        state_batch = torch.cat(batch.state).to(self.device)  # 상태 캣으로 결합
        # action_batch = torch.cat(batch.action)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(self.device)  # 액션 캣으로 결합
        # print(action_batch)
        reward_batch = torch.cat(batch.reward).to(self.device)  # 보상 캣으로 결합

        state_action_values = self.model(state_batch).gather(1, action_batch).to(self.device)  # 모델에서 상태 액션 값 계산
        next_state_values = torch.zeros(self.batch_size).to(self.device)  # 다음 상태 값 초기화
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach().to(self.device)  # 다음 상태 값 계산
        expected_state_action_values = reward_batch + self.gamma * next_state_values  # 기대 상태 액션 값 계산
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1)).to(self.device)  # 손실 계산
        self.optimizer.zero_grad()  # 옵티마이저 초기화
        loss.backward()  # 역전파
        self.optimizer.step()  # 옵티마이저 업데이트

    def save_memory(self, state, action, next_state, reward):  # 메모리에 저장
        self.memory.push(state, action, next_state, reward)  # 메모리에 저장

    def save_model(self, id=0):  # 모델 저장
        torch.save(self.model.state_dict(), f"./models/model{id}.pth")  # 모델 저장

    def load_model(self, id=0):  # 모델 불러오기
        try:
            if id == 0:  # id가 0일 때
                file_names = os.listdir("./models")  # 모델 파일 리스트
                numbers = [re.findall(r"\d+", file_name)[0] for file_name in file_names if re.findall(r"\d+", file_name)]  # 숫자만 추출
                id = max(numbers)  # 최대값
            print(f"./models/model{id}.pth")  # 경로 출력
            self.model.load_state_dict(torch.load(f"./models/model{id}.pth", map_location=self.device, weights_only=True))  # 모델 불러오기
            # print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model file not found. Please check the path.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise ValueError("Model is not loaded")
