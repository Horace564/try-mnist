import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn as nn 
import time


lr = 0.001
batch_size = 512
epoch = 10
transform = transforms.Compose([transforms.ToTensor()])


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64,10)

    def forward(self,x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.log_softmax(self.fc4(x),dim=1)
        return x
    
def get_data(is_train):
    data = MNIST("", train = is_train, transform=transform, download=True)
    return DataLoader(data, batch_size=batch_size, shuffle=True)

def train(train_data, net, optimizer):
    for inputs,lables in train_data:
        optimizer.zero_grad()
        outputs = net(inputs.view(-1,28*28))
        loss = nn.functional.nll_loss(outputs, lables)
        loss.backward()
        optimizer.step()

def test(test_data, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,lables in test_data:
            outputs = net(inputs.view(-1,28*28))
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




