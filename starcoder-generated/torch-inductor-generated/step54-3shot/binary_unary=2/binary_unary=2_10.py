
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 1, stride=1)
    def forward(self, X0):
        v1 = self.conv1(X0)
        v2 = v1[17, 0, 0, 0]
        v3 = v1[42, 1, 0, 0]
        v4 = v2
        v5 = v4 - v3
        return v5

# Inputs to the model
X0 = torch.randn(1, 3, 64, 64)
