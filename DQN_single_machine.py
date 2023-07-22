import collections
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
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
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

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 2)
        else:
            return out.argmax().item()
    
    def select_action(self, obs):
        out = self.forward(obs)
        return out.argmax().item()
    
class Single_machine():
    def __init__(self, df):
        self.x = [0, 0, 0]
        self.stop = 0
        self.df = df
        self.jobs_info = {i: self.df['job' + str(i)] for i in range(1, 101)}
        self.act = []
        self.seq = [i for i in range(1, 101)]
        self.memo = {}

    def step(self, a):
        select_job = self.dispatching_act(a)
        self.act.append(select_job)
        self.stop += 1
        reward = - self.get_fitness(self.act)[0]
        total_flowtime = self.get_fitness(self.act)[1]
        done = self.is_done()
        self.x = [total_flowtime, self.stop, 100-self.stop]
        return self.x, reward, done

    def dispatching_act(self, a):
        if a == 0:  # SPT
            self.seq.sort(key=lambda x: self.jobs_info[x]['소요시간'])
        elif a == 1:  # EDD
            self.seq.sort(key=lambda x: self.jobs_info[x]['제출기한'])
        else:  # slack time
            self.seq.sort(key=lambda x: self.jobs_info[x]['제출기한'] - self.jobs_info[x]['소요시간'])
        chosen_job = self.seq.pop(0)
        return chosen_job
    
    # def get_fitness(self,sequence):
    #     flowtime = 0
    #     total_flowtime = 0
    #     tardiness = 0
    #     total_tardiness = 0
    #     tardy_job = 0
    #     for i in sequence:
    #         if i == 0:
    #             break
    #         job = self.df['job' + str(i)]
    #         flowtime += job['소요시간']
    #         total_flowtime += flowtime
    #         tardiness = max(flowtime - job['제출기한'], 0)
    #         total_tardiness += tardiness
    #         tardy_job += 1 if tardiness > 0 else 0
    #     return tardiness, total_flowtime

    def get_fitness(self, sequence):
        key = tuple(sequence)  # sequence 리스트를 튜플로 변환하여 키로 활용
        if key in self.memo:
            return self.memo[key]
        flowtime = 0
        total_flowtime = 0
        tardiness = 0
        for i in sequence:
            if i == 0:
                break
            job = self.df['job' + str(i)]
            flowtime += job['소요시간']
            total_flowtime += flowtime
            tardiness = max(flowtime - job['제출기한'], 0)
        result = (tardiness, total_flowtime)
        self.memo[key] = result
        return result
    
    def is_done(self):
        if self.stop == 100:
            return True
        else: 
            return False

    def reset(self):
        self.stop = 0
        self.x= [0, 0, 0] # [ 그때 state의 total_flowtime, 인덱스 위치, 남은 job 수 ]
        self.act = []
        self.seq = [i for i in range(1, 101)]
        return self.x
    
def train(q, q_target, memory, optimizer):
    for i in range(10): # 1번의 업데이트에 32개의 데이터가 사용됨, 한 에피소드가 끝날 때마다 버퍼에서 총 320개의 데이터를 뽑아서 사용함
        s, a, r, s_prime, done_mask = memory.sample(batch_size) # 리플레이 버퍼에서 미니 배치 추출
        q_out = q(s)
        q_a = q_out.gather(1, a) # 실제 선택된 액션의 q값을 의미 # a에 해당하는 q값 추출
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) # q_target 네트워크는 정답지를 계산할 떄 쓰이는 네트워크로 학습 대상이 아니다.
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target) # 손실 함수인 MSE 계산
        optimizer.zero_grad() # 현재 배치에 대한 그라디언트를 계산하기 전에 모델의 매개변수 그라디언트를 초기화하는 역할
        loss.backward() # loss에 대한 그라디언트 계산이 일어남
        optimizer.step() # Qnet의 파라미터의 업데이트가 일어남

def main():
    df = pd.read_csv('100_job_uniform data.csv', index_col=0)
    env = Single_machine(df)
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(num_episodes):
        epsilon = max(0.01, 0.08-0.03*(n_epi/200))
        s = env.reset()
        s = np.array(s)
        done = False
        score = 0.0

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

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(n_epi, score, memory.size(), epsilon*100))
        
    s = env.reset()
    s = np.array(s)
    done = False
    score = 0.0
    while not done:
        a = q.select_action(torch.from_numpy(s).float())
        s_prime, r, done = env.step(a)
        s = np.array(s)
        s_prime = np.array(s_prime)
        s = s_prime
        score += r
        if done:
            break
    return score

if __name__ == '__main__':
    score = main()    
    print('Score', score)

    