
import torch.nn as nn
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 15, 1, 0)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.relu(F.pad(t1, (0, 0, 0, 0, 0, 0, 0, 0)))
        return t2.relu()
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
