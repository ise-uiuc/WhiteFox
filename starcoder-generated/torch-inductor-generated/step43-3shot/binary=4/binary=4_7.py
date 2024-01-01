
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1, __other__):
        v1 = self.linear(x1)
        v2 = v1 + __other__
        return v2

# Initializing the model and random input tensor
m = Model()
x1 = torch.randn(4, 16)
__other__ = torch.randn(4, 32)

# Outputs of the model
v3 = m(x1, __other__)

