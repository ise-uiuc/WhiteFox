
import torch
from torch import nn
class SinkCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 5)
        self.linear2 = nn.Linear(5, 2)
    def forward(self, x):
        x = self.linear1(x)
        x = x * x
        x = self.linear2(x).view(-1)
        assert(x.shape == torch.Size([2]))
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
