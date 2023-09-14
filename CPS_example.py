import pandas as pd

########################################################################################################
# (a) 고객 13명의 출발시각을 구하여라.
Arrival_Times = [12, 40, 80, 95, 103, 154, 200, 270, 304, 346, 450, 480, 496]
Service_Times = [30, 60, 55, 48,  33,  50,  20,  74,  28,  21,  47,  35,  12]
data = pd.DataFrame({'Arrival_Times': Arrival_Times, 'Service_Times': Service_Times}).T
data = data.rename(columns={i: f'Customer {i+1}' for i in range(data.shape[1])})
departure_times = []

for i in range(data.shape[1]):
    arrival_time = data.iloc[0, i]
    service_time = data.iloc[1, i]
    
    if i == 0 or arrival_time >= departure_times[i - 1]:
        start_time = arrival_time
    else:
        start_time = departure_times[i - 1]
    
    departure_time = start_time + service_time
    departure_times.append(departure_time)

# 결과를 출력
# 결과값 
result = pd.DataFrame({'Customer': data.columns, 'Departure_Times': departure_times})
print("1번 정답 : " ,result)
print('-----------------------------------------------')

########################################################################################################
# (b) Server가 1대 더 있을 때의 고객 13명의 출발시각을 구하여라.
Arrival_Times = [12, 40, 80, 95, 103, 154, 200, 270, 304, 346, 450, 480, 496]
Service_Times = [30, 60, 55, 48,  33,  50,  20,  74,  28,  21,  47,  35,  12]
data = pd.DataFrame({'Arrival_Times': Arrival_Times, 'Service_Times': Service_Times}).T
data = data.rename(columns={i: f'Customer {i+1}' for i in range(data.shape[1])})

num_servers = 2
departure_times = [0] * data.shape[1]

for i in range(data.shape[1]):
    arrival_time = data.iloc[0, i]
    service_time = data.iloc[1, i]
    
    for server in range(num_servers):
        if departure_times[server] <= arrival_time:
            break
    
    start_time = max(arrival_time, departure_times[server])
    departure_time = start_time + service_time
    
    departure_times[i] = departure_time

result = pd.DataFrame({'Customer': data.columns, 'Departure_Times': departure_times})
print("2번 정답 : " ,result)
print('-----------------------------------------------')
########################################################################################################
# (c) Server가 대기열의 처리방식을 가장 조금 기다린 고객부터 처리하는 식으로 방침을 바꿨을 때의 고객 13명의 출발시각을 구하여라.(Server는 여전히 1대)
Arrival_Times = [12, 40, 80, 95, 103, 154, 200, 270, 304, 346, 450, 480, 496]
Service_Times = [30, 60, 55, 48,  33,  50,  20,  74,  28,  21,  47,  35,  12]

data = pd.DataFrame({'Arrival_Times': Arrival_Times, 'Service_Times': Service_Times})

departure_times = []
current_time = 0
while not data.empty:
    eligible_customers = data[data['Arrival_Times'] <= current_time]
    if eligible_customers.empty:
        current_time = data['Arrival_Times'].min()
    else:
        next_customer = eligible_customers['Arrival_Times'].idxmax()
        arrival_time = data.loc[next_customer, 'Arrival_Times']
        service_time = data.loc[next_customer, 'Service_Times']

        if arrival_time > current_time:
            current_time = arrival_time

        departure_time = current_time + service_time
        departure_times.append(departure_time)

        data = data.drop(next_customer)
        current_time = departure_time

result = pd.DataFrame({'Customer': [f'Customer {i+1}' for i in range(len(departure_times))],'Departure_Time': departure_times})
print("3번 정답 : " ,result)
