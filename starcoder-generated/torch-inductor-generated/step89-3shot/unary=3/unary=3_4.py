
import torch.nn as nn
import math
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 1, 6, 1, 0),
            nn.Conv2d(1, 2, 5, 1, 0),
            nn.BatchNorm2d(3),
            nn.Conv2d(2, 5, 3, 2, 0)
        )
        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose2d(5, 1, 3, 2, 0),
            nn.Conv2d(1, 5, 10, 1, 0),
            nn.ReLU(True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    def forward(self, x1):
        v1 = self.conv_block1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = v6 + self.conv_block2(x1)
        return v7
# Inputs to the model
x1 = torch.randn(2, 3, 123, 123)
