import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # 2차원이미지를 하기위해 Conv2d사용, (in_channels : 흑백1, 컬러3, out_channels : 필터 갯수, kernel_size : 필터 크기, padding) 
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128) # pooling을 거친 결과 32개의 채널을 갖게 되고, 각 채널의 크기가 7x7
        self.fc2 = nn.Linear(128, 10) # output : 0~10 숫자

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # convolution 연산 후 activation 연산 후 pooling 연산
        x = self.pool(torch.relu(self.conv2(x))) # 같은 작업 두 번
        x = torch.flatten(x, 1) # flattening 작업
        x = torch.relu(self.fc1(x)) # flattening 아웃풋이 신경망에 input으로 들어감
        x = self.fc2(x)
        return x

def main():
    # 이미지 변환들을 순차적으로 적용할 수 있도록 해주는 함수(transforms.Compose)
    # transforms.ToTensor() : PIL 이미지나 numpy 배열을 Tensor로 변환시켜줌, 원래 이미지의 픽셀이 [0,255] 범위이면 [0,1]로 정규화해줌
    # transforms.Normalize((0.5,), (0.5,)) : 흑백 이미지는 평균과 표준 편차를 사용하여 정규화 수행, 이미지의 픽셀 값들이 -1~1 사이의 범위로 다시 조정, 학습의 안정화와 빠른 수렴에 도움을 줌
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # batch_size : 한 번에 처리할 미니배치의 크기를 지정, 4개의 이미지와 레이블을 한 번에 처리
    # shffule : True는 매 에폭마다 데이터를 섞어서 미니배치를 만듬
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

    net = SimpleCNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
    print("Finished Training")
    test(net, testloader)

def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on the test images: {100 * correct / total}%")

if __name__ == '__main__':
    main()


    
