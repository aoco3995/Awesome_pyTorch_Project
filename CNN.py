import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    """
    A convolutional neural network (CNN) for image classification.
    
    This CNN class inherits from PyTorch's nn.Module class and
    defines the following layers:
        - conv1: a 2D convolutional layer with in_channels, out_channels, kernel_size, stride, and padding
        - conv2: a 2D convolutional layer with in_channels and out_channels equal to 10, kernel_size, stride, and padding
        - conv3: a 2D convolutional layer with in_channels and out_channels equal to 10, kernel_size, stride, and padding
        - pool: a max pooling layer with kernel_size and stride
        - fc1: a fully connected layer with 10 * 25 * 25 input features and 120 output features
        - fc2: a fully connected layer with 120 input features and 80 output features
        - fc3: a fully connected layer with 80 input features and 40 output features
        - fc4: a fully connected layer with 40 input features and 5 output features
        - dropout: a dropout layer with drop probability 0.1
        
    The forward() method applies the above layers in sequence, with ReLU activation functions
    applied after the convolutional and fully connected layers, and with max pooling applied
    after the first, second, and third convolutional layers.

    Args:
        in_channels (int): the number of input channels for the first convolutional layer
    """
    def __init__(self,in_channels):
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
        """
        Forward pass of the CNN.
        
        This method applies the layers defined in the __init__() method in sequence,
        passing the input through the convolutional and fully connected layers and
        applying ReLU activation functions and max pooling as defined in the __init__() method.
        
        Args:
            x (torch.Tensor): the input tensor of shape (batch_size)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 10 * 25 * 25)
        X = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x