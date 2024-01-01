
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = nn.ModuleList(
            [nn.ConvTranspose2d(4, 4, 3, stride=1, padding=1),
             nn.ConvTranspose1d(10, 3, 3, stride=1, padding=1)]
        )
    def forward(self, x1):
        v1 = self.conv_transpose[0](x1)
        v1 = torch.tanh(v1)
        v2 = self.conv_transpose[1](v1)
        v2 = torch.tanh(v2)
        return v2
# Inputs to the model
x1 = torch.randn(4, 4, 9, 9)
x2 = torch.randn(1, 13, 22)
