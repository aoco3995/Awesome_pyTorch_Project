import torch
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
print(device)

# Hyperparameters
in_channels = 3
num_classes = 5
learning_rate = 2e-6
batch_size = 4
num_epochs = 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
train_percent = 0.9

# Load Data
dataset = projectDataset(csv_file = 'data/project2Dataset.csv', img_dir='data/project2Dataset',transform=None)
#train_set = projectDataset(csv_file = 'data/project2Dataset.csv', img_dir='data/project2Dataset',transform=transforms.ToTensor())
#test_set = projectDataset(csv_file='data/test_set.csv', img_dir='data/test_set', transform=transforms.ToTensor())
train_size = int(train_percent*len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
classes = ('pikachu', 'drone', 'dog', 'cat', 'people')
# torchvision.datasets.CIFAR10.url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=0)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=0)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
#dataiter = iter(train_loader)
#images, labels = next(dataiter)

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))




#model = torchvision.models.resnet50()
model = CNN(in_channels)
model.to(device)

trainer = Trainer(model, device, learning_rate, num_epochs, train_loader)
trainer.train()
trainer.save()


def check_accuracy(loader, model):
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

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.3f}%')

    #model.train()

def check_many(loader, model):
        # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for images, x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

print("Checking accuracy on Training Set")
check_many(train_loader, model)

print("Checking Accuracy on Test Set")
check_many(test_loader, model)

