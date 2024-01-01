
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 24, 7, stride=2, padding=3, dilation=1, groups=1)
        self.conv2 = torch.nn.Conv2d(24, 24, 1, groups=1)
        self.conv3 = torch.nn.Conv2d(36, 36, 1, groups=1, padding=1, stride=2)
        self.conv4 = torch.nn.Conv2d(24, 33, 7, padding=1, stride=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(torch.cat([v4,v1],1))
        v6 = torch.tanh(v5)
        return self.conv4(v6)
# Inputs to the model
import torch
x = torch.randn(1, 1, 56, 224)
