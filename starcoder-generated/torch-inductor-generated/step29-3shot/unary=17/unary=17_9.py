
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(3, 3, kernel_size=(2, 2), bias=False),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        )
    def forward(self, x):
        x = self.block(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 16, 16)
