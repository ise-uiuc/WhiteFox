
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(15984, 8064, 3, stride=1, padding=0)
    def forward(self, x10):
        v1 = self.conv(x10)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
# Inputs to the model
x10 = torch.randn(1, 15984, 1, 3)

import torch
from torch import nn, optim
import torch.nn.functional as F

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 4, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 4, 3, stride=1, padding=1)
    def forward(self, x5):
        v1 = self.conv1(x5)
        v2 = F.max_pool2d(v1, 2, 2, 0)
        v3 = self.conv2(v2)
        # v4 = F.max_pool2d(v3, 2, 2, 0)
        v5 = self.conv3(v3)
        v6 = F.max_pool2d(v5, 2, 2, 0
        )
        v7 = v6.reshape(-1, 64)
        return v7

        # Inputs to the model
x5 = torch.randn(1, 3, 128, 128)
