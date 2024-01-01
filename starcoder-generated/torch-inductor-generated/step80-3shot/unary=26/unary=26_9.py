
import numpy as np
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 12, 3, stride=1, padding=1, bias=False)
    def forward(self, x8):
        t1 = self.conv_t(x8)
        t2 = t1 > 0
        t3 = t1 * -0.1285
        t4 = torch.where(t2, t4, t3)
        return torch.nn.functional.relu(t4)
x8 = torch.randn(63, 16, 18, 17)
