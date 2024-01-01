
import torch
import torch.nn as nn
from torch.optim import Adam

class ModelTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 3, 1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(64, 6, 32, 32)
