
import torch
import  torch.nn
from   torch.nn import  Conv2d
layer = Conv2d(3, 3, 3, stride=1, padding=1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(3, 3, 3, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3
model = Model()
model(x1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
