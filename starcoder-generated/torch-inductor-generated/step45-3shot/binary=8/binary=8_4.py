, which doesn't contain the desired pattern
import torch
nn = torch.nn
F = torch.nn.functional
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = F.conv2d(x1, x2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 512, 8, 8)
x2 = torch.randn(1, 1, 512, 1)
