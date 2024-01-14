# Action은 왼쪽으로 이동 (a=0), 오른쪽으로 이동(a=1)
# Action 6번하면 Episode 종료
# Action을 (0, 1, 0, 1, 0, 0) 순서로 취했을 때 R = +1000
# Reward는 오른쪽은 +1, 왼쪽은 -1
# (1, 1, 1, 1, 1, 1)은 local optimal : 나오면 안되는 경우

####################################
#      #      #      #      #      #
#      #      # 시작 #      #      #
#      #      #      #      #      #
####################################

import numpy as np
import random
import copy

class LR_world():
    def __init__(self):
        self.x = []

    def step(self, a):
        if a == 0:
            if self.x == [0, 1, 0, 1, 0]:
                reward = +1000
            else:
                reward = -1
            self.move_left()
        else:
            reward = +1
            self.move_right()

        done = self.is_done()
        return self.x, reward, done

    def move_left(self):
        self.x.append(0)

    def move_right(self):
        self.x.append(1)

    def is_done(self):
        if len(self.x) == 6:
            return True
        else: 
            return False
        
    def reset(self):
        self.x = []
        return self.x

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((127,2))       
        self.eps = 0.9
        self.alpha = 0.01

    # def state(self, s):
    #     state = 0
    #     if len(s) == 0:
    #         state = 1
    #     else:
    #         state += int("".join([str(bit) for bit in s]), 2)
    #     return state
    
    def state(self, s):
        state = 0
        for bit in s:
            state = state * 2 + bit
        if len(s) > 0:
            state += 2 ** (len(s) + 1)
        return state

    def select_action(self, s):
        k = self.state(s)
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,1)
        else:
            action_val = self.q_table[k,:]
            action = np.argmax(action_val)
        return action

    def select_bestaction(self, s):
        x = self.state(s)
        action_val = self.q_table[x,:]
        action = np.argmax(action_val)
        return action

    def update_table(self, history):
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x = self.state(s)
            cum_reward = cum_reward + r
            self.q_table[x, a] = self.q_table[x, a] + self.alpha * (cum_reward - self.q_table[x, a])
        return cum_reward

    def anneal_eps(self):
        self.eps -= 0.001
        self.eps = max(self.eps, 0.2)

    def show_table(self):
        q_lst = self.q_table.tolist()
        #print(q_lst)
        
def main():
    env = LR_world()
    agent = QAgent()
    best_score = -float('inf')
    best_epi = []

    for n_epi in range(15000):
        done = False
        history = []
        score = 0.0

        s = env.reset()
        while not done:
            s = s[:]
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((copy.deepcopy(s), a, r, copy.deepcopy(s_prime)))
            s = s_prime
            score += r

        agent.update_table(history) 
        agent.anneal_eps()

    #     if score == 999.0:
    #         best_epi.append(n_epi)

    #     if n_epi%9==0:
    #         print("n_episode : {}, score : {:.1f}".format(n_epi, score))
    #         agent.show_table()

    #     if score >= best_score:
    #         best_table = []
    #         best_score = score
    #         best_table = agent.q_table.tolist()

    # print("\nBest table score : {:.1f}, best_episode 갯수: {}".format(best_score, len(best_epi)))
    # print('Best table :', best_table)

    done=False
    s=env.reset()
    total_reward = 0
    while not done:
        s = s[:]
        a = agent.select_bestaction(s)
        s_prime, r, done = env.step(a)
        s_prime = s_prime[:]
        total_reward = total_reward + r
        s = s_prime

    #print(s,total_reward)
    
    return total_reward

average = 0
for i in range(100):
    total_reward = main()
    print(i+1 , "회 최적정책 리워드는 ", total_reward)
    average = total_reward + average

print(average/100, "은 평균")