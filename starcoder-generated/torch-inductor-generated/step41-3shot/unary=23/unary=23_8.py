
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(6, 2, 3, stride=2)
    def forward(self, x1, x2):
        v1 = torch.tanh(x1 * x2) + torch.sigmoid(x1 * x2)
        v2 = self.conv_transpose(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.sigmoid(x1 * x2) + torch.tanh(x1 * x2)
        v5 = torch.tanh(v4)
        v6 = torch.tanh(v5 * v6) + torch.sigmoid(v4 * v6)
        v7 = v1 - v2
        v8 = v1 * v3
        return v7 * v8
# Inputs to the model
x1 = torch.randn(1, 6, 28)
x2 = torch.randn(1, 6, 28)
