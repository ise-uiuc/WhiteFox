
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv2(x)
        v3 = v1 + v2
        v4 = torch.tanh(v3)
        v5 = self.conv3(v3)
        v6 = self.conv4(v3)
        v7 = v5 + v6
        v8 = torch.tanh(v7)
        v9 = self.conv5(v7)
        v10 = torch.relu(x)
        v11 = v9 + v10
        return v11
# Inputs to the model
import numpy as np
x1 = torch.randn(1, 16, 64, 64)
