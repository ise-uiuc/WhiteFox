
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 2, padding=1, stride=1, bias=False)
        self.conv2 = torch.nn.Conv2d(6, 3, 1, padding=0, stride=1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = torch.nn.functional.interpolate(v2, scale_factor=2, mode="nearest")
        v4 = v3 + v2
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
