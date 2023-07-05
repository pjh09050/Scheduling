import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
import time
import datetime

objective_list = ['total_flowtime', 'makespan', 'tardy_job', 'total_tardiness', 'total_weighted_tardiness']

params = {
    'MUT': 0.2,  # 변이확률
    'END' : 0.95,  # 설정한 비율만큼 sequence가 수렴하면 탐색을 멈추게 하는 파라미터
    'POP_SIZE' : 100,  # population size 10 ~ 100
    'NUM_OFFSPRING' : 15, # 한 세대에 발생하는 자식 chromosome의 수
    'CHANGE' : 10, # 다음 세대로 가는 자식 교체 수
    'type' : objective_list[3], # 원하는 목적함수
    'num_job' : 100 # job 갯수
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

        result = sequence[:]
        sorted_list = sorted(result)  # 작은 값부터 정렬된 새로운 리스트를 생성합니다.
        for i in range(len(result)):
            index = result.index(sorted_list[i])  # 현재 작은 값의 인덱스를 찾습니다.
            result[index] = i + 1  # 1부터 100까지의 값을 할당합니다.
        
        dup = {x for x in result if result.count(x) > 1}
        if dup:
            print('중복 생김', dup)

        for i in result:
            job = df['job' + str(int(i))]
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

        ob_list = {'total_flowtime':total_flowtime, 'makespan':makespan, 'tardy_job':tardy_job,'total_tardiness':total_tardiness, 'total_weighted_tardiness':total_weighted_tardiness}
        return ob_list[self.params['type']]
    
    def print_average_fitness(self, population):
        population_average_fitness = 0
        for i in range(len(population)):
            population_average_fitness += population[i][1]
        population_average_fitness = population_average_fitness / len(population)
        #print("population 평균 fitness: {}".format(population_average_fitness))
        return population_average_fitness
    
    def sort_population(self, population):
        population.sort(key=lambda x:x[1], reverse = False)
        return population
    
    # def sort_arithmetical(self, population):


    def selection_operater(self, population):
        # 토너먼트 선택(t보다 작으면 두 염색체 중 품질이 좋은 것을 선택)
        # parents_list = []
        # t = 0.7
        # for i in range(2):
        #     sample = random.sample(population, 2)
        #     sample = self.sort_population(sample)
        #     rand = random.uniform(0,1)
        #     if rand < t :
        #         parents_list.append(sample[0][0])
        #     else:
        #         parents_list.append(sample[1][0])
        # return parents_list[0], parents_list[1]

        # 룰렛휠
        parents_list = []
        k_pressure = 2
        roulette_list = [(population[-1][1] - population[i][1]) + (population[-1][1] - population[0][1])/(k_pressure-1) for i in range(len(population))]

        for _ in range(2):
            r = random.uniform(0, sum(roulette_list))
            for i, roullette in enumerate(roulette_list):
                r -= roullette
                if r <= 0:
                    parents_list.append(population[i])
                    break
                if len(parents_list) == 2 and parents_list[0] != parents_list[1]:
                    break   
        return parents_list[0][0], parents_list[1][0]

        # 순위 기반
        # parents_list = []
        # n = len(population)
        # rank_list = [(population[0][1] + (i - 1)*(population[-1][1] - population[0][1])/(n - 1)) for i in range(n)]
        # for _ in range(2):
        #     r = random.uniform(rank_list[0], sum(rank_list))
        #     for i, rank in enumerate(rank_list):
        #         r -= rank
        #         if r <= 0:
        #             parents_list.append(population[i])
        #         if len(parents_list) == 2 and parents_list[0] != parents_list[1]:
        #             break
        # return parents_list[0][0], parents_list[1][0]

    def crossover_operater(self, mom_cho, dad_cho):
        # 순서 교차
        # mom_ch = list(mom_cho)
        # dad_ch = list(dad_cho)
        # offspring_cho = []
        # k1, k2 = sorted(random.sample(range(len(mom_ch)), 2))
        # if k1 == 0:
        #     offspring_cho.extend(mom_ch[k1:k2])
        #     offspring_cho.extend([x for x in dad_ch[k2:] + dad_ch[:k2] if x not in mom_ch[k1:k2]])
        # else:
        #     offspring_cho.extend([x for x in dad_ch[k2:] + dad_ch[:k2] if x not in mom_ch[k1:k2]][:k1])
        #     offspring_cho.extend(mom_ch[k1:k2])
        #     offspring_cho.extend([x for x in dad_ch[k2:] + dad_ch[:k2] if x not in offspring_cho])
        # return offspring_cho
    
        # PMX 교차
        # mom_ch = list(mom_cho)
        # dad_ch = list(dad_cho)
        # offspring_cho = [0] * len(mom_ch)
        # k1, k2 = sorted(random.sample(range(len(mom_ch)), 2))
        # offspring_cho[k1:k2] = mom_ch[k1:k2]
        # for i in range(len(mom_ch)):
        #     if i < k1 or i >= k2:
        #         gene = dad_ch[i]
        #         while gene in mom_ch[k1:k2]:
        #             index = mom_ch.index(gene)
        #             gene = dad_ch[index]
        #         offspring_cho[i] = gene
        # return offspring_cho
    
        # 싸이클 교차
        # mom_ch = list(mom_cho)
        # dad_ch = list(dad_cho)
        # offspring_cho = [None] * len(mom_ch)
        # index = 0
        # while None in offspring_cho:
        #     if offspring_cho[index] is None:
        #         if index % 2 == 0:
        #             value = mom_ch[index]
        #             while value not in offspring_cho:
        #                 offspring_cho[index] = value
        #                 index = dad_ch.index(value)
        #                 value = mom_ch[index]
        #         else:
        #             value = dad_ch[index]
        #             while value not in offspring_cho:
        #                 offspring_cho[index] = value
        #                 index = mom_ch.index(value)
        #                 value = dad_ch[index]
        #     index += 1
        # return offspring_cho

        # 평균값 연산
        # mom_ch = list(mom_cho)
        # dad_ch = list(dad_cho)
        # offspring_cho = []
        # for i in range(len(mom_ch)):
        #     avg_gene = (mom_ch[i] + dad_ch[i]) / 2
        #     offspring_cho.append(avg_gene)
        # return offspring_cho
        
        # 부모에서 랜덤하게 선택
        # mom_ch = list(mom_cho)
        # dad_ch = list(dad_cho)
        # offspring_cho = []
        # for i in range(len(mom_ch)):
        #     random_gene = random.choice([mom_ch[i], dad_ch[i]])
        #     offspring_cho.append(random_gene)
        # return offspring_cho

        # 부모에서 교차하여 선택
        # mom_ch = list(mom_cho)
        # dad_ch = list(dad_cho)
        # offspring_cho = []
        # for i in range(len(mom_ch)):
        #     if i % 2 == 0:
        #         offspring_cho.append(mom_ch[i])
        #     else:
        #         offspring_cho.append(dad_ch[i])
        # return offspring_cho
    
        # 부모의 가중 평균으로 자식 선택
        mom_ch = list(mom_cho)
        dad_ch = list(dad_cho)
        offspring_cho = []
        weight = 0.6
        for i in range(len(mom_ch)):
            if i % 2 == 0:
                weighted_avg_gene = weight * mom_ch[i] + (1-weight) * dad_ch[i]
            else:
                weighted_avg_gene = (1-weight) * mom_ch[i] + weight * dad_ch[i]
            offspring_cho.append(weighted_avg_gene)
        return offspring_cho

        # 일점 교차
        # offspring_cho = []
        # mom_ch = list(mom_cho)
        # dad_ch = list(dad_cho)
        # k = random.randint(0, len(mom_ch))
        # offspring_cho.extend(mom_ch[:k])
        # offspring_cho.extend(dad_ch[k:])
        # return offspring_cho

    def mutation_operater(self, chromosome):
        # exchange mutation
        # ex_mu1, ex_mu2 = sorted(random.sample(range(self.params['num_job']), 2)) 
        # chromosome[ex_mu1],chromosome[ex_mu2] = chromosome[ex_mu2], chromosome[ex_mu1]
        # return chromosome

        # 2 exchange mutation
        ex_mu1, ex_mu2 = sorted(random.sample(range(self.params['num_job']), 2))
        chromosome[ex_mu1],chromosome[ex_mu2] = chromosome[ex_mu2], chromosome[ex_mu1]
        ex_mu3, ex_mu4 = sorted(random.sample(range(self.params['num_job']), 2)) 
        chromosome[ex_mu3],chromosome[ex_mu4] = chromosome[ex_mu4], chromosome[ex_mu3]
        return chromosome

        # scramble mutation
        # sc_mu1, sc_mu2 = sorted(random.sample(range(self.params['num_job']), 2))
        # scramble_list = chromosome[sc_mu1:sc_mu2]
        # random.shuffle(scramble_list)
        # chromosome[sc_mu1:sc_mu2] = scramble_list
        # return chromosome

        # 랜덤 갯수 랜덤 변경
        # random_select = random.randrange(5,30) # 변경할 갯수 정하기
        # select = random.sample(range(self.params['num_job']), random_select)
        # for i in select:
        #     chromosome[i] = random.uniform(0,1)
        # return chromosome
        
        # 한 번 섞기
        # random.shuffle(chromosome)
        # return chromosome

        # 확률적으로 변경
        # mutation_rate = 0.2
        # for i in range(len(chromosome)):
        #     if random.uniform(0,1) < mutation_rate:
        #         new_value = random.uniform(0,1)
        #         chromosome[i] = new_value
        # return chromosome

    def replacement_operator(self, population, offsprings):
        result_population = []
        population = self.sort_population(population)
        offsprings = self.sort_population(offsprings)
        # 자식해 집단 중 좋은 자식들만 해집단 중 가장 안좋은 해 대체
        offsprings = offsprings[:self.params["CHANGE"]]
        for i in range(len(offsprings)):
            population[-(i+1)] = offsprings[i]
        result_population = self.sort_population(population)
        return result_population
    
    # 해 탐색(GA) 함수
    def search(self):
        time_list = []
        start = time.time()
        generation = 0  # 현재 세대 수
        population = [] # 해집단
        offsprings = [] # 자식해집단
        average = []

        # 1. 초기화: 랜덤하게 해를 초기화
        for i in range(self.params["POP_SIZE"]):
            chromosome = [(x - 1) / (self.params['num_job'] - 1) for x in range(1, self.params['num_job'] + 1)]
            #chromosome = list(range(1, self.params['num_job']+1))
            random.shuffle(chromosome)
            fitness = self.get_fitness(chromosome)
            population.append([chromosome, fitness])
            population = self.sort_population(population)
        #print(f"minimize {self.params['type']} initialzed population : \n", population, "\n\n")

        while 1:
            offsprings = []
            for i in range(self.params["NUM_OFFSPRING"]):
                # 2. 선택 연산
                mom_ch, dad_ch = self.selection_operater(population)

                # 3. 교차 연산
                offspring = self.crossover_operater(mom_ch, dad_ch)

                # 4. 변이 연산
                if random.uniform(0,1) <= self.params["MUT"]:
                    offspring = self.mutation_operater(offspring)
                fitness = self.get_fitness(offspring)
                offsprings.append([offspring,fitness])

            # 5. 대치 연산
            population = self.replacement_operator(population, offsprings)
            generation += 1
            average.append(population[0][1])
            #self.print_average_fitness(population) # population의 평균 fitness를 출력함으로써 수렴하는 모습을 보기 위한 기능

            if generation%100 == 0:
                print("{} generation population 최소 fitness: {}".format(generation,population[0][1]))

            # 6. 알고리즘 종료 조건 판단
            end = time.time()
            sec = end - start
            time_list.append(sec)
            # 10분 지나면 종료
            # if  sec + time_list[0] >= 600:
            #     #print("탐색이 완료되었습니다. \t 최종 세대수: {},\t 최종 해: {},\t 최종 적합도: {}".format(generation, population[0][0], population[0][1]))
            #     result = population[0][0][:]
            #     sorted_list = sorted(result)  
            #     for i in range(len(result)):
            #         index = result.index(sorted_list[i])  
            #         result[index] = i + 1  
            #     dup = {x for x in result if result.count(x) > 1}
            #     if dup:
            #         print('중복 생김', dup)
            #     print("탐색이 완료되었습니다. \t 최종 세대수: {},\t 최종 해: {},\t 최종 적합도: {}".format(generation, result, population[0][1]))
            #     result_time = datetime.timedelta(seconds=(sec))
            #     print('소요 시간 :', result_time)
            #     break
            # END비율만큼 수렴하면 출력
            # same = 0
            # for i in range(self.params["POP_SIZE"]):
            #     if population[0][1] == population[i][1]:
            #         same += 1
            # if same >= len(population) * self.params["END"] and a == True: 
            #     print("탐색이 완료되었습니다. \t 최종 세대수: {},\t 최종 해: {},\t 최종 적합도: {}".format(generation, population[0][0], population[0][1]))
            #     print('같은 해 갯수 : ', same)
            #     result_time = datetime.timedelta(seconds=(sec))
            #     print('소요 시간 :', result_time)
            #     a = False
                #break

        plt.figure(figsize=(12,6))
        plt.plot(average)
        plt.ylim(average[-1]*0.99, average[0]*1.005)
        #plt.show()

if __name__ == "__main__":
    input_data = pd.read_csv('100_job_uniform data.csv', index_col=0)
    df = input_data
    ga = GA_scheduling(params)
    ga.search()