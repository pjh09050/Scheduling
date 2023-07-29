import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# many to one using LSTM

class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True) 
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 32).to(x.device)
        c0 = torch.zeros(1, x.size(0), 32).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out


def main():
    # 시계열 데이터 생성
    seq_length = 100
    sequence = np.sin(np.linspace(0, 2 * np.pi, seq_length))

    # 시퀀스 데이터를 window 단위로 나누기
    window_size = 10 # 이전 10개의 시점까지 본다
    windows = []
    for i in range(len(sequence) - window_size):
        windows.append(sequence[i : i + window_size + 1])
    data_windows = np.array(windows)

    # 데이터를 PyTorch 텐서로 변환
    input_data = torch.tensor(data_windows[:, :-1], dtype=torch.float32).view(-1, window_size, 1)
    target_data = torch.tensor(data_windows[:, -1], dtype=torch.float32).view(-1, 1)

    model = SimpleLSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 1000
    for epoch in range(num_epochs):
        output = model(input_data)
        loss = loss_function(output, target_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 새로운 시퀀스 생성
    future = 50
    with torch.no_grad():
        current_seq = sequence[-window_size:].tolist()
        predicted_seq = []
        for _ in range(future):
            input_seq = torch.tensor(current_seq[-window_size:], dtype=torch.float32).view(1, window_size, 1)
            pred = model(input_seq)
            current_seq.append(pred.item())
            predicted_seq.append(pred.item())

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.title('Simple LSTM - Sequence Generation')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)
    plt.plot(np.arange(seq_length), sequence, label='Original', color='b')
    plt.plot(np.arange(seq_length, seq_length + future), predicted_seq, label='Generated', color='r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
