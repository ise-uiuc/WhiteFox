
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, padding=2, bias=True)
        self.max = max
        self.min = min
    def forward(self, x1):
        v1 = F.conv2d(x1, self.conv.weight, bias=self.conv.bias, groups=self.conv.groups, padding=self.conv.padding, dilation=self.conv.dilation, stride=self.conv.stride)
        v2 = torch.clamp_max(v1, self.max)
        v3 = torch.clamp_min(v2, self.min)
        return v3
min  = -1.0
max = 0.9
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
