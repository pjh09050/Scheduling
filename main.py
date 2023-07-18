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

learning_rate = 0.0005
gamma = 1
buffer_limit = 50000
batch_size = 32
num_episodes = 1000

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, result_lst = [], [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition # done_mask : 종료 상태의 밸류를 마스킹해주기 위해 만든 변수
            s_lst.append(s) # 스칼라 값이 아닌 벡터 값이 나올 수 있기 때문
            a_lst.append([a])
            r_lst.append([r]) 
            s_prime_lst.append(s_prime) # 스칼라 값이 아닌 벡터 값이 나올 수 있기 때문
            done_mask_lst.append([done_mask]) 
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

# Qnet
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 100)
        self.remaining_jobs = list(range(1, 101))
        self.pre_jobs = []

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def reset(self):
        self.remaining_jobs = list(range(1, 101))
        self.pre_jobs = []

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            action = random.choice(self.remaining_jobs)
            self.remaining_jobs.remove(action)
            self.pre_jobs.append(action)
            return action
        else:
            q_values = out.detach().numpy()
            for action_index in self.pre_jobs:
                q_values[action_index-1] = float('-inf')
            max_action = np.argmax(q_values)
            max_action += 1
            self.remaining_jobs.remove(max_action)
            self.pre_jobs.append(max_action)
            return max_action
    
    def select_action(self, obs):
        out = self.forward(obs)
        q_values = out.detach().numpy()
        for action_index in self.pre_jobs:
            q_values[action_index-1] = float('-inf')
        max_action = np.argmax(q_values)
        max_action += 1
        self.remaining_jobs.remove(max_action)
        self.pre_jobs.append(max_action)
        return max_action            

class Single_machine():
    def __init__(self, df):
        self.x = [0 for i in range(100)]
        self.stop = 0
        self.df = df

    def step(self, a):
        self.x[self.stop] = a
        self.stop += 1
        reward = - self.get_fitness(self.x)
        done = self.is_done()
        self.prev_action = a
        return self.x, reward, done

    def is_done(self):
        if self.stop == 100:
            return True
        else: 
            return False

    def reset(self):
        self.stop = 0
        self.x= [0 for _ in range(100)]
        return self.x
    
    def get_fitness(self,sequence):
        flowtime = 0
        total_flowtime = 0
        makespan = 0
        tardiness = 0
        total_tardiness = 0
        tardy_job = 0
        total_weighted_tardiness = 0

        for i in sequence:
            if i == 0:
                break
            job = self.df['job' + str(i)]
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
        return tardiness

# 학습 함수        
def train(q, q_target, memory, optimizer):
    for i in range(10): # 1번의 업데이트에 32개의 데이터가 사용됨, 한 에피소드가 끝날 때마다 버퍼에서 총 320개의 데이터를 뽑아서 사용함
        s, a, r, s_prime, done_mask = memory.sample(batch_size) # 리플레이 버퍼에서 미니 배치 추출
        q_out = q(s)
        q_a = q_out.gather(1, a-1) # 실제 선택된 액션의 q값을 의미 # a에 해당하는 q값 추출
        ad_q_target = q_target(s_prime)
        for j in range(len(q_target(s_prime))):
            ad_s = s[j].tolist()
            ad_s = [int(x) for x in ad_s if int(x)!=0 ]
            for action_index in ad_s:
                ad_q_target[j][action_index-1] = float('-inf')
        max_q_prime = ad_q_target.max(1)[0].unsqueeze(1) # q_target 네트워크는 정답지를 계산할 떄 쓰이는 네트워크로 학습 대상이 아니다.
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target) # 손실 함수 정의
        optimizer.zero_grad()
        loss.backward() # loss에 대한 그라디언트 계산이 일어남
        optimizer.step() # Qnet의 파라미터의 업데이트가 일어남

def main():
    df = pd.read_csv('100_job_uniform data.csv', index_col=0)
    env = Single_machine(df)
    q = Qnet() # 학습
    q_target = Qnet() # 정답을 계산
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(num_episodes):
        epsilon = max(0.01, 0.08-0.03*(n_epi/200))
        s = env.reset()
        q.reset()
        s = np.array(s)
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done = env.step(a)
            s = np.array(s)
            s_prime = np.array(s_prime)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            score += r
            if done:
                break

        if memory.size() > 2000:  # 데이터가 충분히 쌓이지 않은 상태에서 학습을 진행하면 초기의 데이터가 많이 재사용되어 학습이 치우칠 수 있다.
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict()) # q네트워크의 파라미터를 q_target네트워크로 복사
            print("n_episode : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score, memory.size(), epsilon*100))
        score = 0.0

    s = env.reset()
    q.reset()
    s = np.array(s)
    done = False
    while not done:
        a = q.select_action(torch.from_numpy(s).float())
        s_prime, r, done = env.step(a)
        s = np.array(s)
        s_prime = np.array(s_prime)
        done_mask = 0.0 if done else 1.0
        memory.put((s, a, r, s_prime, done_mask))
        s = s_prime
        score += r
        if done:
            break

if __name__ == '__main__':
    main()    
    