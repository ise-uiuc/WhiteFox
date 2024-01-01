
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Conv3d(32, 32, 3, padding=[1,1,1])

    def forward(self, x1):
        v1 = self.qkv(x1)
        v2 = v1.norm(p=2, dim=1, keepdim=True)
        v3 = v2
        v4 = (v3*v3).sum(dim=1, keepdim=True)
        v5 = v4.pow(-1)
        v6 = v5.transpose(-2,-1)*v3
        v7 = v6.softmax(dim=1)
        v8 = v7.matmul(v1)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 7, 7, 7)
