import pandas as pd
########################################################################################################
# (a) 고객 13명의 출발시각을 구하여라.
Arrival_Times = [12, 40, 80, 95, 103, 154, 200, 270, 304, 346, 450, 480, 496]
Service_Times = [30, 60, 55, 48,  33,  50,  20,  74,  28,  21,  47,  35,  12]
data = pd.DataFrame({'Arrival_Times': Arrival_Times, 'Service_Times': Service_Times}).T

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
print('(a)답:', departure_times)
########################################################################################################
# (b) Server가 1대 더 있을 때의 고객 13명의 출발시각을 구하여라.
Arrival_Times = [12, 40, 80, 95, 103, 154, 200, 270, 304, 346, 450, 480, 496]
Service_Times = [30, 60, 55, 48, 33, 50, 20, 74, 28, 21, 47, 35, 12]
data = pd.DataFrame({'Arrival_Times': Arrival_Times, 'Service_Times': Service_Times}).T

num_servers = 2 
departure_times = [[] for _ in range(num_servers)]
departure_times_list = []
for i in range(data.shape[1]):
    arrival_time = data.iloc[0, i]
    service_time = data.iloc[1, i]

    # 두 대의 기계 중에서 더 빨리 끝나는 기계 선택
    selected_server = min(range(num_servers), key=lambda server: departure_times[server][-1] if departure_times[server] else arrival_time)

    if arrival_time >= departure_times[selected_server][-1] if departure_times[selected_server] else arrival_time:
        start_time = arrival_time
    else:
        start_time = departure_times[selected_server][-1]

    departure_time = start_time + service_time
    departure_times[selected_server].append(departure_time)
    departure_times_list.append(departure_time)
print('(b)답:',departure_times_list)
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
print('(c)답:', departure_times)