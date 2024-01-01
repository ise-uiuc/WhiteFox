
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=2)
        for i in range(3):
            if torch.sum(x) < 0:
                x = torch.pow(x, 2)
            else:
                x = torch.nn.functional.relu(x)
        y = torch.unsqueeze(x, dim=1)
        z = torch.squeeze(y, dim=1)
        return z
# Inputs to the model
x = torch.randn(2, 5, 3, 3)
