
import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 6)
    def forward(self, x):
        x = self.layers(x)
        return torch.cat((x[..., :1] + x[..., 1:2] + x[..., 2:], x), -1)
# Inputs to the model
x = torch.randn(2, 3)
