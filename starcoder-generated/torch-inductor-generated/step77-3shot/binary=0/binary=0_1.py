
import numpy as np
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 2, stride=1, padding=0)
        self.conv1x1 = torch.nn.Conv2d(1, 1, 1, stride=2, bias=False)
    def forward(self, x1):
        var1 = self.conv(x1)
        var1 += torch.ones(1, 1, var1.size(2), 0).cuda()
        var1 = var1.transpose(1, -1)
        var1 = self.conv1x1(var1)
        var1 = var1.permute((0, 1, 3, 2))
        var1 = var1 - torch.FloatTensor(var1.shape).fill_(1).cuda()
        var2 = var1.reshape(var1.size()[0], var1.size()[1] * var1.size()[2])
        var3 = var2.sum(axis=1)
        var4 = var3.reshape(1, var3.size(0), 1)
        return var4
# Inputs to the model
x1 = torch.Tensor(np.load("input0.npz")["arr_0"])
