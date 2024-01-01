
import torch
import torch.nn.functional as func

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        return func.bmm(v1, v2)

# Inputs to the model (x1, x2)
x1 = 6 * torch.randn(4, 3, 5)
x2 = 5 * torch.randn(4, 2, 5)
