
import torch
import torch.nn as nn
import torch.nn.functional as functional
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=2)
        self.sigmoid = functional.sigmoid
        self.linear = nn.Linear(in_features=1, out_features=2)
        self.clamp = torch.clamp
    def forward(self, x):
        x1 = self.clamp(x, 0.0, 10.0)
        x2 = self.linear1(x1)
        x3 = self.sigmoid(x2)
        x4 = x3.permute(0, 2, 1)
        x5 = functional.linear(x4, weight=self.linear.weight, bias=self.linear.bias)
        return x2 + x5
# Inputs to the model
x1 = torch.randn(1, 1, 2)
