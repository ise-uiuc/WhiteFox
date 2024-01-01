
import torch
from torch import nn

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 3, 1, 1)
    def forward(self, x):
        y = x.permute(2, 0, 1).contiguous()
        y = self.conv1(y)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4, 5) # A tensor with dimensionality [2, 3, 4, 5]
