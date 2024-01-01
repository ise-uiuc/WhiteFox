
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

model = torch.nn.DataParallel(
    Model(), device_ids = [0], output_device = [0]).cuda()

# Inputs to the model
# x1 = torch.randn(1, 8, 64, 64).cuda()
x1 = torch.autograd.Variable(torch.rand(1, 3, 64, 64)).cuda()
# 