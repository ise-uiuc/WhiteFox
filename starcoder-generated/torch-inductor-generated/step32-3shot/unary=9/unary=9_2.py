
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 3)
    def forward(self, x1):
        v1 = self.conv(x1.contiguous())
        v2 = 3 + v1
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v3 / 6
        v5 = self.other_conv(v4.contiguous())
        v6 = 3 + v5
        v7 = torch.clamp(v6, min=0, max=6)
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(2, 3, 65, 65)
