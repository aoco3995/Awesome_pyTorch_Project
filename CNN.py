import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self,in_channels):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1))
<<<<<<< HEAD
        #self.fc1 = nn.Linear(10 * 50 * 50, 5)
        self.fc1 = nn.Linear(10 * 50 * 50, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 40)
        ##self.fc4 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(40, 10)
        self.dropout = nn.Dropout(p=0.9) 
=======
        #self.fc1 = nn.Linear(10 * 50 * 50, 120)
        self.fc1 = nn.Linear(10 * 50 * 50, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 48)
        self.fc4 = nn.Linear(48, 24)
        self.fc5 = nn.Linear(24, 5)
>>>>>>> d2089c868149728f0358ecb22fa1de33e9a9ea1d

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 50 * 50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)


        """x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
<<<<<<< HEAD
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(self.fc4(x))
        #x = self.fc1(x)
=======
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.fc1(x)"""
>>>>>>> d2089c868149728f0358ecb22fa1de33e9a9ea1d
        return x