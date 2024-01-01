
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20, 20)
    def forward(self, x):
        x1 = x.permute(0, 2, 1)
        x2 = self.linear(x1)
        x3 = x2.permute(0, 2, 1)
        x4 = x2.mul(x3)
        return x4
# Inputs to the model
x = torch.randn(10, 20, 1)
