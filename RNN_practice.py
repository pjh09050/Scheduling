import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 시퀀스 데이터 생성 함수
def generate_sequence(seq_length):
    return np.sin(np.linspace(0, 2 * np.pi, seq_length))

# 시퀀스 데이터를 window 단위로 나누는 함수
def create_window_data(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i : i + window_size + 1])
    return np.array(windows)

# 시계열 데이터 생성
seq_length = 100
sequence = generate_sequence(seq_length)

# 시퀀스 데이터를 window 단위로 나누기
window_size = 10
data_windows = create_window_data(sequence, window_size)

# 데이터를 PyTorch 텐서로 변환
input_data = torch.tensor(data_windows[:, :-1], dtype=torch.float32).view(-1, window_size, 1)
target_data = torch.tensor(data_windows[:, -1], dtype=torch.float32).view(-1, 1)

# 간단한 RNN 모델 정의
class SimpleRNN(nn.Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(1, 32, batch_first=True)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 32).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.linear(out[:, -1, :])
        return out

# 모델 초기화
model = SimpleRNN()

# 손실 함수와 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# 훈련
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target_data)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 새로운 시퀀스 생성
with torch.no_grad():
    future = 50
    current_seq = sequence[-window_size:].tolist()
    predicted_seq = []
    for _ in range(future):
        input_seq = torch.tensor(current_seq[-window_size:], dtype=torch.float32).view(1, window_size, 1)
        pred = model(input_seq)
        current_seq.append(pred.item())
        predicted_seq.append(pred.item())

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.title('Simple RNN - Sequence Generation')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.grid(True)
plt.plot(np.arange(seq_length), sequence, label='Original', color='b')
plt.plot(np.arange(seq_length, seq_length + future), predicted_seq, label='Generated', color='r')
plt.legend()
plt.show()
