
import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        b1 = self.conv1(x1)
        b2 = F.max_pool2d(b1, kernel_size=3, stride=2, padding=1)
        b3 = F.relu(b2)
        b4 = self.conv2(b3)
        b5 = b4 + x2
        b6 = torch.sigmoid(b5)
        b7 = self.conv3(b6)
        b8 = b7 + x3
        b9 = torch.sigmoid(b8)
        return b9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 64, 64, 64)
x3 = torch.randn(1, 64, 64, 64)
