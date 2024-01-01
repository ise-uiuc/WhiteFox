
import torch.randn
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(1,1), stride=(1,1), bias=True)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x
x = torch.randn(16,1,48,256)
