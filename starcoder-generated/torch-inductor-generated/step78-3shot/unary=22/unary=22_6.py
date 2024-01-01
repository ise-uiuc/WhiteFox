
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 11)

    def forward(self, x):
        return F.tanh(self.linear(x))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 7)
