
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, 1)
        self.fc = nn.Linear(8828, 10)

    # def forward(self, x, bias, weight): # PyTorch 1.8
    def forward(self, x, weight, bias): 
        # weight = nn.Parameter(weight)
        # bias = nn.Parameter(bias)

        # x = F.conv2d(x, weight, padding=1)
        x = F.conv2d(x, weight, bias, padding=1)
        return x

# Inputs to the model, note that the first tensor
# is the output of torch.randn(1, 3, 28, 28)
x = torch.randn(1, 3, 28, 28)
weight = torch.randn(8, 3, 3, 3)
bias = torch.randn(1, 8, 1, 1)
