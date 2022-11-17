import torch
import torch.nn as nn
import torch.optim as optim
import logging


class Trainer():
    def __init__(self, model, device, learning_rate, num_epochs, train_loader):
        self.num_epochs =num_epochs
        self.train_loader = train_loader
        self.device = device
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        

    def train(self):

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
                if i % 200 == 199:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                    running_loss = 0.0

            print(f'Cost at epoch {epoch+1} is {sum(losses)/len(losses)}')
            logging.info('Fininshed Set')
        print('Finished Training')

    def save(self):
        PATH = './custom_classifier_dataset.pth'
        torch.save(self.model.state_dict(), PATH)

        