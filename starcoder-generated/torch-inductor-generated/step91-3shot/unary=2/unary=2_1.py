
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 2, bias=False, dilation=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        temp1 = v1.transpose(1, 2).transpose(2, 1)
        v2 = temp1 * 0.5
        v3 = temp1 * temp1 * temp1
        v4 = v3 * 0.044715
        temp2 = v1 + v4
        v5 = temp2 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v6 + 1
        v8 = v2 * v7
        v9 = v1.sum(axis=2, keepdim=True).sum(axis=1, keepdim=True)
        return v9
# Inputs to the model
x1 = torch.randn(3, 1, 6, 6)
