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
import pickle

class CPU_Unpickler(pickle.Unpickler):
    """A class for unpickling PyTorch objects on the CPU.

    This class extends the functionality of the built-in `pickle.Unpickler`
    class by providing a custom `find_class` method that allows for loading
    PyTorch storage objects from bytes on the CPU.

    Args:
        pickle.Unpickler: The base unpickler class from the `pickle` module.

    Attributes:
        None

    Methods:
        find_class: Overrides the base `find_class` method to provide custom
            behavior for loading PyTorch storage objects from bytes on the CPU.
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameters
in_channels = 3
num_classes = 5
learning_rate = 1e-8#27e-4
batch_size = 4
num_epochs = 96
train_percent = 0.9
train_seed = 2
momentum = 0.4
weight_decay = 0.2
dampening = 0.2

# Load Data
dataset = projectDataset(csv_file = 'data/project2Dataset.csv', img_dir='data/project2Dataset',transform=None)

train_size = int(train_percent*len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(train_seed))

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

classes = ('pikachu', 'drone', 'dog', 'cat', 'person')




# functions to show an image


def imshow(img):
    """
    imshow(img)

    Displays an image using matplotlib.

    This function takes in an image tensor `img` and displays it using matplotlib.
    The image tensor is first unnormalized by applying the formula `img = img / 2 + 0.5`,
    then converted to a NumPy array using the `numpy()` method, and finally displayed
    using `plt.imshow()`, with the dimensions of the image tensor permuted to the correct
    order using `np.transpose()`.

    """
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

trainer = Trainer(model, device, learning_rate, num_epochs, train_loader, momentum, weight_decay, dampening)
if input("Load[y/n]:  ") == "y":
    PATH = './custom_classifier_dataset.pth'

    with open(PATH, "rb") as f:
        contents = CPU_Unpickler(f).load()

    model.load_state_dict(torch.load(contents), map_location=torch.device('cpu'))
    model.eval()

if input("Train[y/n]:  ") == "y":
    trainer.train()





def check_accuracy(loader, model):
    """
    check_accuracy(loader, model)

    Displays an accuracy rating of the model given the a specific data set.

    This function takes in an dataset and displays it using the terminal.
    """
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

#print("Checking Costs accoss epochs")
#trainer.cost_list()

if input("Save[y/n]:  ") == "y":
    trainer.save()

