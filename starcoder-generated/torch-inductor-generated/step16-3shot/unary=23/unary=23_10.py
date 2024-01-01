
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose2d(8, 8, kernel_size=(2, 2), bias=False, padding=(1, 1))
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v3 = self.bn(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
