import torch

class TrainingAccuracy():
    def __init__(self):

        self.correct = 0
        self.total = 0

    def accuracy(self,trainer,testloader):
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = trainer.outputs(images)
                _, predicted = torch.max(outputs.data, 1)
                self.total += labels.size(0)
                self.correct += (predicted == labels).sum().item()

    def outputs(self):

        print('Accuracy of the network on the 10000 test images: %d %%' % (
               100 * self.correct / self.total))

    #test