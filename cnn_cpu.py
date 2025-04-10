import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn as nn 
import time

lr = 0.001
batch_size = 512
epoch = 10
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]) #known data(mean,standard deviation) about MNIST dataset
criterion = nn.CrossEntropyLoss()

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,10,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,20,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) 
        )
        self.fc1 = nn.Linear(320,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self,x):
        #demension of input x: (batch_size, channel_size, height, width)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x.view(-1,320))
        x = self.fc2(x)
        return x
    

def get_data(is_train):
    data = MNIST("", train = is_train, transform=transform, download=True)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def train(train_data, net, optimizer):
    for inputs,lables in train_data:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()


def test(test_data, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,lables in test_data:
            outputs = net(inputs)
            for i,output in enumerate(outputs):
                if torch.argmax(output) == lables[i]:
                    correct+=1
                total+=1
    return correct/total

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


def main():
    start_time = time.time()
    train_data = get_data(is_train=True)
    test_data  = get_data(is_train=False)
    print(f"[epoch 0], accurancy is {test(test_data,net)*100:.2f}%")
    
    for i in range(epoch):
        train(train_data,net,optimizer)
        print(f"[epoch {i+1}], accurancy is {test(test_data,net)*100:.2f}%")
    print("Total training time(CPU) is ",round(time.time()-start_time, 2), "s")
    
if __name__ == '__main__':
    main()
