
import torch
class Test(torch.nn.Module):
    def forward(self, x):
        t = torch.rand_like(x)
        return t
# Inputs to the model
x1 = torch.randn(1, 2)
