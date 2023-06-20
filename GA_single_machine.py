import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

params = {
    'MUT': 0.1,  # 변이확률
    'END' : 0.9,  # 설정한 비율만큼 sequence가 수렴하면 탐색을 멈추게 하는 파라미터
    'POP_SIZE' : 100,  # population size 10 ~ 100
    'NUM_OFFSPRING' : 10, # 한 세대에 발생하는 자식 chromosome의 수
    'CHANGE' : 3, # 다음 세대로 가는 자식 교체 수
    'type' : 'total_flowtime', # 원하는 목적함수
    'num_job' : 10 # job 갯수
    }
# ------------------------------

class GA_scheduling():
    def __init__(self, parameters):
        self.params = {}
        for key, value in parameters.items():
            self.params[key] = value

    def get_fitness(self, sequence):
        flowtime = 0
        total_flowtime = 0
        makespan = 0
        tardiness = 0
        total_tardiness = 0
        tardy_job = 0
        total_weighted_tardiness = 0
        
        for i in sequence:
            job = df['job' + str(i)]
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

        return total_flowtime, makespan, tardy_job, total_tardiness, total_weighted_tardiness
    
    def print_average_fitness(self, population):
        population_average_fitness = 0
        for i in range(len(population)):
            population_average_fitness += population[i][1]
        population_average_fitness = population_average_fitness / len(population)
        print("population 평균 fitness: {}".format(population_average_fitness))
        return population_average_fitness
    
    def sort_population(self, population):
        population.sort(key=lambda x:x[1])
        return population

    def selection_operater(self, population):
        mom_ch = 0
        dad_ch = 0
        # 토너먼트 선택(t보다 작으면 두 염색체 중 품질이 좋은 것을 선택)
        t = 0.6
        for i in range(2):
            sample = random.sample(population, 2)
            sample = self.sort_population(sample)
            rand = random.uniform(0,1)
            if i == 0:
                if rand < t:
                    mom_ch = sample[0][0]
                else:
                    mom_ch = sample[1][0]
            if i == 1:
                if rand < t:
                    dad_ch = sample[0][0]
                else:
                    dad_ch = sample[1][0]

        return mom_ch, dad_ch

    def crossover_operater(self, mom_cho, dad_cho):
        # 순서 교차
        mom_ch = list(mom_cho)
        dad_ch = list(dad_cho)
        offspring_cho = [0] * self.params['num_job']
        k1 = random.randint(0, len(mom_ch)/2-1)
        k2 = random.randint(len(mom_ch)/2, len(mom_ch)-1) 
        offspring_cho[k1:k2] = mom_ch[k1:k2]
        index = k2  

        for i in range(len(dad_ch)):
            if dad_ch[index] not in offspring_cho:
                offspring_cho[offspring_cho.index(0)] = dad_ch[index]
                index += 1
            else:
                index += 1
            if index == len(offspring_cho):
                index = 0

        return offspring_cho

    def mutation_operater(self, chromosome):
        # exchange mutation
        # ex_mu1 = random.randint(0, self.params['num_job']-1)
        # ex_mu2 = random.randint(0, self.params['num_job']-1)
        # while ex_mu1 == ex_mu2:
        #     ex_mu2 = random.randint(0, self.params['num_job']-1)
        # chromosome[ex_mu1],chromosome[ex_mu2] = chromosome[ex_mu2], chromosome[ex_mu1]

        # scramble mutation
        sc_mu1 = random.randint(0, self.params['num_job'] - 3)
        sc_mu2 = random.randint(sc_mu1 + 1, self.params['num_job'] - 1)
        while sc_mu2 - sc_mu1 < 2:
            sc_mu1 = random.randint(0, self.params['num_job'] - 3)
            sc_mu2 = random.randint(sc_mu1 + 1, self.params['num_job'] - 1)
        scramble_list = chromosome[sc_mu1:sc_mu2]
        random.shuffle(scramble_list)
        chromosome[sc_mu1:sc_mu2] = scramble_list
        return chromosome

    def replacement_operator(self, population, offsprings):
        result_population = []
        population = self.sort_population(population)
        # 자식해 집단 중 뽑고 싶은 자식 수를 파라미터로 받아 가장 안좋은 해 대체
        offsprings = random.sample(offsprings, self.params["CHANGE"])
        for i in range(len(offsprings)):
            population[-(i+1)] = offsprings[i]
        result_population = self.sort_population(population)
        
        return result_population
    
    # 해 탐색(GA) 함수
    def search(self):
        generation = 0  # 현재 세대 수
        population = [] # 해집단
        offsprings = [] # 자식해집단
        average = []

        # 1. 초기화: 랜덤하게 해를 초기화
        for i in range(self.params["POP_SIZE"]):
            chromosome = list(range(1, self.params['num_job']+1))
            random.shuffle(chromosome)
            if self.params['type'] == 'total_flowtime':
                results = self.get_fitness(chromosome)
                fitness = results[0]
            elif self.params['type'] == 'makespan':
                results = self.get_fitness(chromosome)
                fitness = results[1]
            elif self.params['type'] == 'number of tardy jobs':
                results = self.get_fitness(chromosome)
                fitness = results[2]
            elif self.params['type'] == 'total_tardiness':
                results = self.get_fitness(chromosome)
                fitness = results[3]
            elif self.params['type'] == 'total_weighted_tardiness':
                results = self.get_fitness(chromosome)
                fitness = results[4]
            population.append([chromosome, fitness])
            population = self.sort_population(population)
        print(f"minimize {self.params['type']} initialzed population : \n", population, "\n\n")

        while 1:
            offsprings = []
            for i in range(self.params["NUM_OFFSPRING"]):
                            
                # 2. 선택 연산
                mom_ch, dad_ch = self.selection_operater(population)
                
                # 3. 교차 연산
                offspring = self.crossover_operater(mom_ch, dad_ch)

                # 4. 변이 연산
                # todo: 변이 연산여부를 결정, self.params["MUT"]에 따라 변이가 결정되지 않으면 변이연산 수행하지 않음
                if random.uniform(0,1) <= self.params["MUT"]:
                    offspring = self.mutation_operater(offspring)

                if self.params['type'] == 'total_flowtime':
                    results = self.get_fitness(offspring)
                    fitness = results[0]
                elif self.params['type'] == 'makespan':
                    results = self.get_fitness(offspring)
                    fitness = results[1]
                elif self.params['type'] == 'number of tardy jobs':
                    results = self.get_fitness(offspring)
                    fitness = results[2]
                elif self.params['type'] == 'total_tardiness':
                    results = self.get_fitness(offspring)
                    fitness = results[3]
                elif self.params['type'] == 'total_weighted_tardiness':
                    results = self.get_fitness(offspring)
                    fitness = results[4]

                offsprings.append([offspring,fitness])
                
            # 5. 대치 연산
            population = self.replacement_operator(population, offsprings)
            generation += 1

            self.print_average_fitness(population) # population의 평균 fitness를 출력함으로써 수렴하는 모습을 보기 위한 기능
            average.append(self.print_average_fitness(population)) # population의 평균 fitness 그래프를 그리기 위한 average에 추가
            average.append(population[0][1])

            # 6. 알고리즘 종료 조건 판단
            same = 0
            for i in range(self.params["POP_SIZE"]):
                if population[0][1] == population[i][1]:
                    same += 1
            if same >= len(population) * self.params["END"]: # END비율만큼 수렴하면 정지
                break

        # 최종적으로 얼마나 소요되었는지의 세대수, 수렴된 chromosome과 fitness를 출력
        print("탐색이 완료되었습니다. \t 최종 세대수: {},\t 최종 해: {},\t 최종 적합도: {}".format(generation, population[0][0], population[0][1]))
        print('최종 population :', population)
        # population의 평균 fitness 그래프
        plt.plot(average)
        plt.ylim(average[0], average[-1])
        plt.show()

if __name__ == "__main__":
    input_data = pd.read_csv('10_job_normal data.csv', index_col=0)
    df = input_data
    ga = GA_scheduling(params)
    ga.search()