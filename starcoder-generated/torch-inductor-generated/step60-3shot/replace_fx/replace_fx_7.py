
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = x1 + torch.rand_like(x1)
        x3 = x2.detach() - torch.rand_like(x2)
        return x3 * 0.3
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x2 = x1 + torch.rand_like(x1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
