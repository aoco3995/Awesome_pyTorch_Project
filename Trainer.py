import torch
import torch.nn as nn
import torch.optim as optim


class Trainer():

    """Trainer for a PyTorch model.

    Args:
        model (torch.nn.Module): The model to train.
        device (torch.device): The device to train on (e.g. "cpu" or "cuda").
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train for.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        momentum (float): Momentum for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        dampening (float): Dampening for the optimizer.
    """

    def __init__(self, model, device, learning_rate, num_epochs, train_loader, momentum, weight_decay, dampening):
        self.num_epochs =num_epochs
        self.train_loader = train_loader
        self.device = device
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, dampening = dampening)
        self.cost = []
        

    def train(self):

        """Trains the model for the specified number of epochs."""

        for epoch in range(self.num_epochs):  # loop over the dataset multiple times

            losses = []
            running_loss = 0.0
            for i, (_, data, targets) in enumerate(self.train_loader):
                # Get data to cuda if possible
                data = data.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device)

                # forward
                scores = self.model(data)
                loss = self.criterion(scores, targets)

                losses.append(loss.item())

                # backward
                self.optimizer.zero_grad()
                loss.backward()

                # gradient decent
                self.optimizer.step()

                running_loss += loss.item()
                if i % len(data) == len(data)-1:
                    #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(data):.3f}')
                    running_loss = 0.0
            print(f'Cost at epoch {epoch+1} is {sum(losses)/len(losses)}')
            self.cost.append(sum(losses)/len(losses))
        print('Finished Training')

    def save(self):
        """Saves the trained model."""
        PATH = './custom_classifier_dataset.pth'
        torch.save(self.model.state_dict(), PATH)

    def cost_list(self):
        """Prints the cost at each epoch during training."""
        for i in range(len(self.cost)):
            print(f'Cost at epoch {i+1} is {self.cost[i]}')