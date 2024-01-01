
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Conv1d(8, 8, 1, stride=1, padding=1, groups=8)
        self.m2 = torch.nn.Conv1d(8, 8, 1, stride=1, padding=1, padding_mode="circular", groups=8)
    def forward(self, x1):
        v1 = self.m1(x1)
        v2 = self.m2(x1)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(1, 8, 64)
