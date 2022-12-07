import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self,in_channels=3):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1))

        self.fc1 = nn.Linear(10 * 25 * 25, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 40)
        self.fc4 = nn.Linear(40, 5)

        self.dropout = nn.Dropout(p=0.1) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 10 * 25 * 25)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x