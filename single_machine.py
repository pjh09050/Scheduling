class Single_machine():
    def __init__(self, df):
        self.x = [0, 0, 0]
        self.stop = 0
        self.df = df

    def step(self, a):
        if a == 0:
            seq = self.SPT()
        elif a == 1:
            seq = self.EDD()
        else:
            seq = self.SLACK()
        self.stop += 1
        reward = - self.get_fitness(seq)
        flow_time = self.get_flowtime(seq)
        done = self.is_done()
        self.x = [, self.stop, 100-self.stop]
        return self.x, reward, done

    def SPT(df):
        p_time_sort = df.sort_values(by='소요시간', axis=1)
        flowtime = 0
        for i in p_time_sort:
            job = df[i]
            flowtime += job['소요시간']
        return flowtime

    def EDD():
        seq = '뭐로할까?'

    def SLACK():
        seq = '뭐로할까?'
    
    def get_fitness(self,sequence):
        flowtime = 0
        total_flowtime = 0
        tardiness = 0
        total_tardiness = 0
        tardy_job = 0

        for i in sequence:
            if i == 0:
                break
            job = self.df['job' + str(i)]
            flowtime += job['소요시간']
            total_flowtime += flowtime
            tardiness = max(flowtime - job['제출기한'], 0)
            total_tardiness += tardiness
            tardy_job += 1 if tardiness > 0 else 0
        return tardiness
    
    def is_done(self):
        if self.stop == 100:
            return True
        else: 
            return False

    def reset(self):
        self.stop = 0
        self.x= [0, 0, 0] # [ 그때 state의 flowtime, 인덱스 위치, 남은 job 수 ]
        return self.x