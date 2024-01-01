
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        n1 = self.conv(x)
        n2 = torch.tanh(n1)
        return n2
# Inputs to the model
x = torch.randn(1, 6, 32, 32)
