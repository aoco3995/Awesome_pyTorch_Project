import torch

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from customDataset import projectDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from CNN import CNN
from Trainer import Trainer

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 3
num_classes = 5
learning_rate = 1e-3
batch_size = 32
num_epochs = 30
train_percent = 0.9
train_seed = 4

# Load Data
dataset = projectDataset(csv_file = 'data/project2Dataset.csv', img_dir='data/project2Dataset',transform=None)
train_size = int(train_percent*len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(train_seed))
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
classes = ('pikachu', 'drone', 'dog', 'cat', 'person')

model = CNN(in_channels)
model.load_state_dict(torch.load('custom_classifier_dataset.pth'))
model.to(device)

def check_accuracy(loader, model):

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for _, x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += (predictions.size(0))

            for y, predictions in zip(y, predictions):
                if y == predictions:
                    correct_pred[classes[y]] += 1
                total_pred[classes[y]] += 1

        
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.3f}%')

print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking Accuracy on Test Set")
check_accuracy(test_loader, model)

