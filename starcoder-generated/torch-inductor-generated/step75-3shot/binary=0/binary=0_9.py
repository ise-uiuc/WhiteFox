
import torch
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
    def forward(self, x1, padding1=None):
        v1 = self.conv(x1)
        if not padding1 is None:
            padding1 = F.max_pool2d(padding1, 2)
        v2 = v1 + padding1
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
p1 = torch.randn(1, 3, 32, 32)
