
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 22, 7, stride=1, padding=3)
        self.conv1 = nn.ConvTranspose2d(22, 15, 7, stride=1, padding=0, output_padding=0)
        self.conv2 = nn.Conv2d(15, 61, 7, stride=1, padding=0)
        self.conv3 = nn.ConvTranspose2d(61, 50, 5, stride=1, padding=3, output_padding=0)
        self.conv4 = nn.Conv2d(62, 50, 3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(62, 10, 3, stride=1, padding=0, output_padding=0)
        self.conv6 = nn.Conv2d(10, 10, 3, stride=1, padding=1)

    def forward(self, x1):
        x = self.conv(x1)
        x = x * 0.5
        x = self.conv1(x)
        x = torch.erf(x)
        x = x + 1
        x1 = x * x1
        x = self.conv2(x1)
        x = x + 0.5
        x1 = x * x
        x1 = F.relu(x1)
        x = self.conv3(x1)
        x = x + 0.7071067811865476
        x = torch.sigmoid(x)
        x = x * x
        x1 = x * x1
        x = self.conv4(x1)
        x = x + 0.5
        x = torch.log(x1)
        x1 = x * x
        x = self.conv5(x1)
        x = torch.tanh(x + 0.5)
        x = x * x
        x = self.conv6(x)
        x = x * 0.7071067811865476
        x = torch.abs(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 1, 61, 19)
