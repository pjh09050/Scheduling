import pandas as pd
import numpy as np
import random
import itertools

def generate_input_data_uniform(num_jobs):
    data = []
    for job_id in range(1, num_jobs + 1):
        while True:
            release_time = 0
            processing_time = int(max(0, np.random.uniform(1, 10)))
            due_date = int(max(release_time + processing_time, np.random.uniform(8, 15)))
            if release_time < processing_time and processing_time < due_date:
                break
        rate = random.randint(1, 3)
        data.append((job_id, release_time, processing_time, due_date, rate))
    return data

def generate_input_data_normal(num_jobs):
    data = []
    for job_id in range(1, num_jobs + 1):
        while True:
            processing_time = int(max(0, np.random.normal(5, 2)))
            release_time = 0
            #release_time = int(max(0, np.random.normal(1, processing_time)) )
            due_date = int(max(release_time+processing_time, np.random.normal(8, 2)))
            if release_time < processing_time and processing_time < due_date:
                break
        rate = random.randint(1, 3)
        data.append((job_id, release_time, processing_time, due_date, rate))
    return data

def get_fitness(sequence):
    flowtime = 0
    total_flowtime = 0
    makespan = 0
    tardiness = 0
    total_tardiness = 0
    tardy_job = 0
    total_weighted_tardiness = 0
    
    for i in sequence:
        job = df['job' + str(i)]
        a = job['출제시간'] + job['소요시간']
        flowtime += a
        total_flowtime += flowtime

        makespan = max(makespan,  flowtime)
        
        if flowtime - job['제출기한'] >= 0:
            tardiness = flowtime - job['제출기한']
            tardy_job += 1
            total_tardiness += tardiness

        weighted_tardiness = job['성적 반영비율'] * tardiness
        total_weighted_tardiness += weighted_tardiness

    return total_flowtime, makespan, tardy_job, total_tardiness, total_weighted_tardiness


def full_enumeration(object):
    print(len(df.columns))
    num_jobs = len(df.columns)
    best_sequence = None
    minimize_objective = float('inf')

    permutations = list(itertools.permutations(range(1, num_jobs + 1), num_jobs))
    print('full_enumerate 갯수 :', len(permutations))

    for sequence in permutations:
        if object == 'total_flowtime':
            results = get_fitness(sequence)
            result = results[0]
        elif object == 'makespan':
            results = get_fitness(sequence)
            result = results[1]
        elif object == 'number of tardy jobs':
            results = get_fitness(sequence)
            result = results[2]
        elif object == 'total_tardiness':
            results = get_fitness(sequence)
            result = results[3]
        elif object == 'total_weighted_tardiness':
            results = get_fitness(sequence)
            result = results[4]

        objective = result

        if objective < minimize_objective:
            minimize_objective = objective
            best_sequence = sequence

    return object, best_sequence, minimize_objective

################################################################################################################
# 만들고 싶은 job 갯수

num_jobs = 3
#output_data = generate_input_data_uniform(num_jobs) # 균일분포로 데이터 생성
output_data = generate_input_data_normal(num_jobs) # 정규분포로 데이터 생성

df = pd.DataFrame(output_data, columns=['job', '출제시간', '소요시간', '제출기한', '성적 반영비율'])
df = df.set_index('job').transpose()
df.columns = ['job' + str(i) for i in range(1, num_jobs + 1)]
df.index = ['출제시간', '소요시간', '제출기한', '성적 반영비율']

filename = f'{num_jobs}_job_data.csv'
df.to_csv(filename, index=True, encoding='utf-8-sig')

print(f"{num_jobs}개 job data 생성.")

input_file = f'{num_jobs}_job_data.csv'
input_data = pd.read_csv(input_file, index_col=0)
df = input_data

################################################################################################################
# 넣어보고 싶은 sequence

results = get_fitness([1,2,3])

print(f'total flowtime : {results[0]}')
print(f'makespan : {results[1]}')
print(f'number of tardy jobs : {results[2]}')
print(f'total_tardiness : {results[3]}')
print(f'total_weighted_tardiness : {results[4]}')

################################################################################################################
# full_enumeration으로 목적함수를 최소화하는 sequence 찾기

object, best_sequence, minimize_objective = full_enumeration('total_flowtime')

print(f'{object} 최소화')
print(f'Best sequence: {best_sequence}')
print(f'minimize {object}: {minimize_objective}')

################################################################################################################
# GA로 찾기