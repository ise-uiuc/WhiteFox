
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 32)

    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.linear(x)
        return x2
# Inputs to the model
x = torch.rand(1, 16)
