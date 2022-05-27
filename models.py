from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        return self.stack(x)
    
class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()

        self.first_conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.second_conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        # self.relu2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*7*7, out_features=2048),
            nn.Linear(in_features=2048, out_features=1024),
            nn.Linear(in_features=1024, out_features=10),
            nn.Softmax(dim=1)
        )

        # self.flatten = nn.Flatten()

        # self.fc1 = nn.Linear(in_features=64*7*7, out_features=2048)
        # self.relu3 = nn.ReLU()
        # self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        # self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(in_features=1024, out_features=10)


        # self.fc1 = nn.Linear(in_features=128*3*3, out_features=1024)
        # self.fc2 = nn.Linear(in_features=1024, out_features=512)
        # self.fc3 = nn.Linear(in_features=512, out_features=10)

        # self.logits = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.pool1(self.relu(x))

        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.pool2(self.relu(x))

        # # x = self.conv3(x)
        # # x = self.bn3(x)
        # # x = self.pool(self.relu(x))
        # # print(x.shape)

        # x = self.flatten(x)
        # #x = self.dropout(self.relu(self.fc1(x)))
        # x = self.relu(self.fc1(x))
        # #x = self.dropout(self.relu(self.fc2(x)))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # return self.logits(x)

        x = self.first_conv_stack(x)
        x = self.second_conv_stack(x)
        return self.linear_stack(x)


class CNN1(nn.Module):
    def __init__(self, in_channels):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
       
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=32*14*14, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.fc3 = nn.Linear(in_features=1204, out_features=10)
        self.logits = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(self.relu(x))

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.logits(x)