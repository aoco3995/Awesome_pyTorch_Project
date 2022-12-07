
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

# Hyperparameters
in_channels = 3
num_classes = 5
learning_rate = 1e-3
batch_size = 4
num_epochs = 10
train_percent = 0.9


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(10 * 50 * 50, 5)
        #self.fc1 = nn.Linear(10 * 50 * 50, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.fc1(x)
        return x

# Load Trained Model

net = CNN()
net.load_state_dict(torch.load("custom_classifier_dataset.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load("custom_classifier_dataset.pth")
model.eval()

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                        ])
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

image = Image.open("data\project2Dataset\pikachu241.jpg")

predict_image(image)