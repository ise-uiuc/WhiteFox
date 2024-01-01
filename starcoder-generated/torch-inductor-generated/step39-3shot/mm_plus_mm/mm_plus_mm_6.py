
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1, x2, x3):
        t2 = torch.mm(x2, x2)
        return torch.mm(x1, t2) + x3
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
