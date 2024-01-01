
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=(1, 1), bias=False)
    def forward():
        pass
x1 = torch.randn(1, 32, 11, 18)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 11, 18)
