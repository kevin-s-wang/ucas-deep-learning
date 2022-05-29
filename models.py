from turtle import forward
from torch import nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.stack(x)

class NeuralNetworkWider(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetworkWider, self).__init__()

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.stack(x)

# class NeuralNetworkDeeper(nn.Module):
#     def __init__(self) -> None:
#         super(NeuralNetworkDeeper, self).__init__()

#         self.stack = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28*28, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#             nn.Softmax(dim=1)
#         )
    
#     def forward(self, x):
#         return self.stack(x)


class NeuralNetworkDeeper(nn.Module):
    def __init__(self):
        super(NeuralNetworkDeeper, self).__init__()
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.stack(x)

class CNNLarger(nn.Module):
    def __init__(self,):
        super(CNNLarger, self).__init__()

        self.first_conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.second_conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.third_conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*3*3, out_features=10),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.first_conv_stack(x)
        x = self.second_conv_stack(x)
        x = self.third_conv_stack(x)
        return self.linear_stack(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.logits = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))

        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.logits(x)
