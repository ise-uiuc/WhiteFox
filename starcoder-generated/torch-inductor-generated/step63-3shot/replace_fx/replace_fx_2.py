
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = torch.nn.Linear(3, 1)
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = self.linear2(x1)
        x4 = self.linear2(x3)
        x5 = self.linear2(x4)
        return x5
# Inputs to the model
x1 = torch.randn(3, 4)
