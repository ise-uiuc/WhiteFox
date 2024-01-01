
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2d_1a = nn.Conv2d(3, 64, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2d_2a = nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2d_2b = nn.Conv2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1),bias=False)
        self.conv2d_3b = nn.Conv2d(128, 276, (3, 3), stride=(1, 1), padding=(1, 1),bias=False)
        self.conv2d_4a = nn.Conv2d(276, 64, (1, 1), stride=(1, 1), bias=False)
        self.conv2d_4b = nn.Conv2d(64, 96, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    def forward(self, x):
        x1 = self.conv2d_1a(x)
        x2 = nn.ConvTranspose2d(3, 64, (3,3),stride=2)
        x3 = self.conv2d_2a(x2)
        x4 = self.conv2d_2b(x3)
        x5 = self.conv2d_3b(x4)
        x6 = self.conv2d_4a(x5)
        x7 = self.conv2d_4b(x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
