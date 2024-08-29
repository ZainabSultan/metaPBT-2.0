
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch import nn
import torch
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import gpytorch
import math
import tqdm

# define neural net

class NeuralNetworkHeart(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetworkHeart, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 50)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        x = self.tanh(self.layer3(x))
        return x

# Instantiate the model
input_dim = 100  # Example input dimension
model = NeuralNetworkHeart(input_dim)

# Print the model
print(model)

