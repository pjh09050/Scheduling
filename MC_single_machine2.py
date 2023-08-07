import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import warnings
warnings.filterwarnings('ignore')

'''
machine : 1개
Job type : A, B, C
Job number : 각 10개, 총 30개
목적함수 : 가산점 + Throughput(생산 갯수) 최대화
is_done : 110초
processing time : A(10), B(20), C(30)
초기 setup : C
Setup 시간 : C->A:5, C->B:5, A->B:10, A->C:10, B->A:5, B->C:10
가산점 : Throughput에 C가 3개 이상있으면 20점
'''
learning_rate = 0.001
gamma = 0.99
buffer_limit = 50000
batch_size = 32
num_episodes = 3000

class Score_Single_machine():
    def __init__(self):
        self.x = [0, 0, 0, 0, 0, 0]
        self.jobs = {'A': 10, 'B': 10, 'C': 10}
        self.job_processing_time = {'A': 10, 'B': 20, 'C': 30}
        self.all_setup_times = {
            'C': {'A': 5, 'B': 5},
            'A': {'B': 10, 'C': 10},
            'B': {'A': 5, 'C': 10}
        }
        self.current_job_type = 'C'
        self.total_jobs = 30
        self.select_job = []
        self.stop = 0
        self.total_processing_time = 0
        self.total_setup_time = 0
        self.setup_changes = 0
        self.final_score = 0

    def step(self, a):
        a = 'A' if a == 0 else "B" if a == 1 else "C"
        if a != self.current_job_type:
            setup_time = self.change_setup(a)
            self.total_setup_time += setup_time
            self.setup_changes += 1
        self.select_job.append(a)

        # processing_time 계산
        processing_time = self.job_processing_time[a]
        self.total_processing_time += processing_time
        self.stop += processing_time

        done = self.is_done()
        # 가산점
        C_num = 10 - self.jobs['C']
        bonus_points = 20 if C_num >= 3 else 0
        number_of_jobs_produced = self.total_jobs - sum(self.jobs.values())
        reward = bonus_points + number_of_jobs_produced
        if done == True:
            self.final_score = number_of_jobs_produced + bonus_points
        else:
            self.jobs[a] -= 1

        self.x = [bonus_points, C_num, sum(self.jobs.values()), self.total_setup_time, self.setup_changes, 100 - self.stop]
        return self.x, reward, done, self.final_score
        
    def change_setup(self, a):
        setup_time = self.all_setup_times[self.current_job_type][a]
        self.stop += setup_time
        self.current_job_type = a
        return setup_time

    def is_done(self):
        if self.stop >= 110:
            return True
        else:
            return False

    def reset(self):
        self.x = [0, 0, 0, 0, 0, 0]
        self.jobs = {'A': 10, 'B': 10, 'C': 10}
        self.current_job_type = 'C'
        self.select_job = []
        self.stop = 0
        self.total_processing_time = 0
        self.total_setup_time = 0
        self.setup_changes = 0
        self.final_score = 0
        return self.x

class MCnet(nn.Module):
    def __init__(self):
        super(MCnet, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,2)
        else:
            return out.argmax().item()
        
    def select_action(self, obs):
        out = self.forward(obs)
        return out.argmax().item()

def main():
    env = Score_Single_machine()
    MC_net = MCnet()

    optimizer = optim.Adam(MC_net.parameters(), lr=learning_rate)
    print_interval = 20

    for n_epi in range(num_episodes):
        epsilon = max(0.01, 0.08-0.01*(n_epi/200))
        s = env.reset()
        s = torch.from_numpy(np.array(s)).float()  # NumPy array를 Tensor로 변환
        done = False

        episode = []  # 에피소드 내에서 (상태, 액션, 보상)을 저장하기 위한 리스트
        while not done:
            a = MC_net.sample_action(s, epsilon)  # 정책 네트워크에서 액션 샘플링
            s_prime, r, done, final_score = env.step(a)
            s_prime = torch.from_numpy(np.array(s_prime)).float()
            episode.append((s, a, r))
            s = s_prime
            if done:
                break
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            optimizer.zero_grad()
            log_prob = torch.log(MC_net(s)[a])  # 선택된 액션의 로그 확률
            loss = -log_prob * G
            loss.backward()
            optimizer.step()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("n_episode : {}, final_score : {}".format(n_epi, final_score))

    s = env.reset()
    s = torch.from_numpy(np.array(s)).float() 
    done = False
    act_set = []
    while not done:
        a = MC_net.select_action(s)
        s_prime, r, done, final_score2 = env.step(a)
        s_prime = torch.from_numpy(np.array(s_prime)).float()
        s = s_prime
        a = 'A' if a == 0 else "B" if a == 1 else "C"
        act_set.append(a)
        if done:
            break
    return act_set, final_score2

if __name__ == '__main__':
    act_set, final_score2 = main()    
    print('Action : {}, Score : {}'.format(act_set, final_score2))