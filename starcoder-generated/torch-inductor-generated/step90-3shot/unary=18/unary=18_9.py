
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = torch.clamp(v1, 0, 20)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
