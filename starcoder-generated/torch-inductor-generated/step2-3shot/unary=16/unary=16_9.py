
import torch
from torch import nn


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.relu()
        return v2

# Initialize the model

m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
