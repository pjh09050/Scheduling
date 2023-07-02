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

    def selection_operater(self, population):
        # 토너먼트 선택(t보다 작으면 두 염색체 중 품질이 좋은 것을 선택)
        parents_list = []
        t = 0.7
        for i in range(2):
            sample = random.sample(population, 2)
            sample = self.sort_population(sample)
            rand = random.uniform(0,1)
            if rand < t :
                parents_list.append(sample[0][0])
            else:
                parents_list.append(sample[1][0])
        return parents_list[0], parents_list[1]

        # 룰렛휠
        # parents_list = []
        # k_pressure = 2
        # roulette_list = [(population[-1][1] - population[i][1]) + (population[-1][1] - population[0][1])/(k_pressure-1) for i in range(len(population))]

        # for _ in range(2):
        #     r = random.uniform(0, sum(roulette_list))
        #     for i, roullette in enumerate(roulette_list):
        #         r -= roullette
        #         if r <= 0:
        #             parents_list.append(population[i])
        #             break
        #     if len(parents_list) == 2 and parents_list[0] != parents_list[1]:
        #         break   
        # return parents_list[0][0], parents_list[1][0]

        # 순위 기반
        # parents_list = []
        # n = len(population)
        # rank_list = [(population[0][1] + (i - 1)*(population[-1][1] - population[0][1])/(n - 1)) for i in range(n)]

        # for _ in range(2):
        #     r = random.uniform(0, sum(rank_list))
        #     for i, rank in enumerate(rank_list):
        #         r -= rank
        #         if r <= 0:
        #             parents_list.append(population[i])
        #     if len(parents_list) == 2 and parents_list[0] != parents_list[1]:
        #         break
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
        mom_ch = list(mom_cho)
        dad_ch = list(dad_cho)
        offspring_cho = [0] * len(mom_ch)
        k1, k2 = sorted(random.sample(range(len(mom_ch)), 2))
        offspring_cho[k1:k2] = mom_ch[k1:k2]
        for i in range(len(mom_ch)):
            if i < k1 or i >= k2:
                gene = dad_ch[i]
                while gene in mom_ch[k1:k2]:
                    index = mom_ch.index(gene)
                    gene = dad_ch[index]
                offspring_cho[i] = gene
        return offspring_cho
    
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

    def mutation_operater(self, chromosome):
        # exchange mutation
        # ex_mu1, ex_mu2 = sorted(random.sample(range(self.params['num_job']), 2)) 
        # chromosome[ex_mu1],chromosome[ex_mu2] = chromosome[ex_mu2], chromosome[ex_mu1]

        # scramble mutation
        sc_mu1, sc_mu2 = sorted(random.sample(range(self.params['num_job']), 2))
        scramble_list = chromosome[sc_mu1:sc_mu2]
        random.shuffle(scramble_list)
        chromosome[sc_mu1:sc_mu2] = scramble_list
        return chromosome

    def replacement_operator(self, population, offsprings):
        result_population = []
        population = self.sort_population(population)
        offsprings = self.sort_population(offsprings)
        # 자식해 집단 중 랜덤으로 고른 자식만 해집단 중 가장 안좋은 해 대체
        # offsprings = random.sample(offsprings, self.params["CHANGE"])
        # 자식해 집단 중 좋은 자식들만 해집단 중 가장 안좋은 해 대체
        offsprings = offsprings[:self.params["CHANGE"]]
        for i in range(len(offsprings)):
            population[-(i+1)] = offsprings[i]
        result_population = self.sort_population(population)

        # 해집단 전체를 비교하여 자식과 가장 가까운 해를 대치하는 방법
        # for offspring in offsprings:
        #     closest_individual = None
        #     min_fitness_diff = float('inf')

        #     for individual in population:
        #         fitness_diff = abs(individual[1] - offspring[1])
        #         if fitness_diff < min_fitness_diff:
        #             min_fitness_diff = fitness_diff
        #             closest_individual = individual

        #     if closest_individual is not None:
        #         population.remove(closest_individual)  
        #         population.append(offspring) 
        # result_population = self.sort_population(population)
        return result_population
    
    # 해 탐색(GA) 함수
    def search(self):
        time_list = []
        start = time.time()
        generation = 0  # 현재 세대 수
        population = [] # 해집단
        offsprings = [] # 자식해집단
        average = []
        count_avg = []
        a = True
        b = True

        # 1. 초기화: 랜덤하게 해를 초기화
        for i in range(self.params["POP_SIZE"]):
            chromosome = list(range(1, self.params['num_job']+1))
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
                # todo: 변이 연산여부를 결정, self.params["MUT"]에 따라 변이가 결정되지 않으면 변이연산 수행하지 않음
                if random.uniform(0,1) <= self.params["MUT"]:
                    offspring = self.mutation_operater(offspring)
                fitness = self.get_fitness(offspring)
                offsprings.append([offspring,fitness])
                
            # 5. 대치 연산
            population = self.replacement_operator(population, offsprings)
            generation += 1
            # self.print_average_fitness(population) # population의 평균 fitness를 출력함으로써 수렴하는 모습을 보기 위한 기능
            #average.append(self.print_average_fitness(population)) # population의 평균 fitness 그래프를 그리기 위한 average에 추가
            #average.append(population[-1][1])
            average.append(population[0][1])

            # 6. 알고리즘 종료 조건 판단
            end = time.time()
            sec = end - start
            time_list.append(sec)
            # 10분 지나면 종료
            if  sec + time_list[0] >= 600:
                print("탐색이 완료되었습니다. \t 최종 세대수: {},\t 최종 해: {},\t 최종 적합도: {}".format(generation, population[0][0], population[0][1]))
                #print('최종 population :', population)
                result_time = datetime.timedelta(seconds=(sec))
                print('소요 시간 :', result_time)
                print('같은 해 갯수 : ', same)
                break
            
            if population[0][1] == 27834 and b == True:
                print("탐색이 완료되었습니다. \t 최종 세대수: {},\t 최종 해: {},\t 최종 적합도: {}".format(generation, population[0][0], population[0][1]))
                #print('최종 population :', population)
                result_time = datetime.timedelta(seconds=(sec))
                print('소요 시간 :', result_time)
                b = False

            # END비율만큼 수렴하면 정지
            same = 0
            for i in range(self.params["POP_SIZE"]):
                if population[0][1] == population[i][1]:
                    same += 1

            if same >= len(population) * self.params["END"] and a == True: 
                # 최종적으로 얼마나 소요되었는지의 세대수, 수렴된 chromosome과 fitness를 출력
                print("탐색이 완료되었습니다. \t 최종 세대수: {},\t 최종 해: {},\t 최종 적합도: {}".format(generation, population[0][0], population[0][1]))
                #print('최종 population :', population)
                print('같은 해 갯수 : ', same)
                result_time = datetime.timedelta(seconds=(sec))
                print('소요 시간 :', result_time)
                a = False
                #break

            # 7. live plot update
            x = np.linspace(0,300,population[0][1])
            y = np.arange(len(x))
            if generation%100 == 0:
                count_avg.append(population[0][1])
                print("{} generation population 최소 fitness: {}".format(generation,population[0][1]))
                # if generation//100 == 1:
                #     plt.ion()
                #     fig = plt.figure(figsize=(12,6))
                #     ax = fig.add_subplot(111)
                #     line1, = ax.plot(x, y)
                #     plt.title('Genetic Algorithm', fontsize=16, fontweight='bold')
                #     plt.xlabel('generation(단위:100)', fontsize=12)
                #     plt.ylabel('fitness', fontsize=12)
                # else: 
                #     line1.set_xdata(np.arange(len(count_avg)))
                #     line1.set_ydata(count_avg)
                #     ax.set_xlim(0, generation//100)
                #     ax.set_ylim(population[0][1]*0.99, average[0]*1.005)
                #     fig.canvas.draw()
                #     fig.canvas.flush_events()
                #     time.sleep(0.1)
        plt.figure(figsize=(12,6))
        plt.plot(average)
        plt.ylim(average[-1]*0.99, average[0]*1.005)
        plt.show()

if __name__ == "__main__":
    input_data = pd.read_csv('100_job_uniform data.csv', index_col=0) # 지금까지 minimize total tardiness 27834
    df = input_data
    ga = GA_scheduling(params)
    ga.search()