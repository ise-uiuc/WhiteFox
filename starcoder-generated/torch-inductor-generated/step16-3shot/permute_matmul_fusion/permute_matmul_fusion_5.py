
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3, x4, x5):
        v1 = x1.permute(0, 2, 1) # x1: (bsize, 1024, 64)
        v2 = x2.permute(2, 1, 0) # x2: (bsize, 64, 10)
        v3 = torch.bmm(v1, v2)
        v4 = v3.permute(2, 1, 0) 
        v5 = v4.permute(1, 0, 2)

        return torch.matmul(x3.permute(0, 2, 1), x4)

model = Model()

x1 = torch.randn(1, 1024, 64)
x2 = torch.randn(1, 64, 10)
x3 = torch.randn(1, 10, 20)
x4 = torch.randn(1, 32, 10)
x5 = torch.randn(1, 5, 3, 3)
model(x1, x2, x3, x4, x5).shape

