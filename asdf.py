import collections # 선입선출의 특성을 갖고 있는 리플레이 버퍼를 쉽게 구현
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

objective_list = ['total_flowtime', 'makespan', 'tardy_job', 'total_tardiness', 'total_weighted_tardiness']

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, input_dim, output_dim, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, learning_rate=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(input_dim, output_dim).to(self.device)
        self.target_model = DQN(input_dim, output_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.memory = []

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.output_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.model(state)
        return q_values.argmax(dim=1).item()

    def train(self):
        if len(self.memory) < batch_size:
            return

        # 데이터를 배치 형태로 변환
        batch = list(zip(*self.memory))
        states = torch.tensor(batch[0], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(batch[3], dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch[4], dtype=torch.bool).unsqueeze(1).to(self.device)

        # 현재 상태에 대한 Q값 계산
        q_values = self.model(states).gather(1, actions)

        # 다음 상태의 최대 Q값 계산
        next_q_values = self.target_model(next_states).max(dim=1)[0].detach()

        # 타겟 Q값 계산
        targets = rewards + self.gamma * next_q_values * (~dones)

        # 손실 계산 및 업데이트
        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 메모리 비우기
        self.memory = []

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

def get_fitness(df, sequence):  # 데이터를 전달받도록 수정
    flowtime = 0
    total_flowtime = 0
    makespan = 0
    tardiness = 0
    total_tardiness = 0
    tardy_job = 0
    total_weighted_tardiness = 0

    for i in sequence:
        job = df['job' + str(int(i) + 1)]
        flowtime += job['소요시간']
        total_flowtime += flowtime

        makespan = flowtime

        if flowtime - job['제출기한'] >= 0:
            tardiness = flowtime - job['제출기한']
            tardy_job += 1
            total_tardiness += tardiness
        else:
            tardiness = 0

        weighted_tardiness = job['성적 반영비율'] * tardiness
        total_weighted_tardiness += weighted_tardiness

    ob_list = {'total_flowtime': total_flowtime, 'makespan': makespan, 'tardy_job': tardy_job,
               'total_tardiness': total_tardiness, 'total_weighted_tardiness': total_weighted_tardiness}
    return ob_list[objective_list[3]]

# 학습 파라미터 설정
input_dim = 100
output_dim = 100
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
learning_rate = 0.001
batch_size = 32

# 데이터 로딩
df = pd.read_csv('100_job_uniform data.csv', index_col=0)

# 에이전트 초기화
agent = DQNAgent(input_dim, output_dim, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate)

# 학습 루프
num_episodes = 1000
max_steps = 100

for episode in range(num_episodes):
    state = np.random.permutation(input_dim)
    total_tardiness = get_fitness(df, state)

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state = np.copy(state)
        next_state[action], next_state[action-1] = next_state[action-1], next_state[action]
        tardiness = get_fitness(df, next_state)
        reward = -tardiness
        total_tardiness += tardiness

        agent.memory.append((state, action, reward, next_state, False))

        state = next_state

    agent.train()
    agent.update_target_model()

    print(f"Episode: {episode+1}/{num_episodes}, Total Tardiness: {total_tardiness}")

# 최종 정책 평가
test_episodes = 10

for episode in range(test_episodes):
    state = np.random.permutation(input_dim)
    total_tardiness = get_fitness(df, state)

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state = np.copy(state)
        next_state[action], next_state[action-1] = next_state[action-1], next_state[action]
        tardiness = get_fitness(df, next_state)
        total_tardiness += tardiness

        state = next_state

    print(f"Test Episode: {episode+1}/{test_episodes}, Total Tardiness: {total_tardiness}")
