import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

class SimpleCNN(nn.Module):
    """Simple CNN class which takes, at initialization, the number of input channels we want from the network. 
    Inherits from the nn.Module class"""

    def __init__(self, n_chans=32):
        super().__init__()
        self.n_chans = n_chans

        #Convolutional layers with random initial kernels
        self.conv1 = nn.Conv2d(
            in_channels=1, #Number of input channels = 1 (B&W)
            out_channels=16, #Outputs to 16 different channels
            kernel_size=3, #3x3 kernel matrix for the convolution
            padding=1 #Padding maintains the original dimensions of the image.
            )
        
        self.conv2 = nn.Conv2d(
            in_channels=16, #Number of input channels = 16 (previous layer)
            out_channels=8, #Outputs to 16 different channels
            kernel_size=3, #3x3 kernel matrix for the convolution
            padding=1 #Padding maintains the original dimensions of the image.
            )
        

        #Fully-connected linear layers
        self.fc1 = nn.Linear(in_features=392, out_features=32) #Reduce features from 512 to 32
        self.fc2 = nn.Linear(in_features=32, out_features=10) #Reduce further from 32 to 10- the number of categories in FashionMNIST

        #Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """Defines the forward pass of the training loop, using two maxpooling layers, two convolutional layers, and two fully-
        connected linear layers, using a tanh activation funtion."""
        out = F.max_pool2d(input=torch.tanh(self.conv1(x)), kernel_size=2) #reduces to 16x16 feature map
        out = F.max_pool2d(input=torch.tanh(self.conv2(out)), kernel_size=2) #reduces to 8x8
        out = out.view(-1, 392) #Unsqueeze into a 392-dim vector
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


