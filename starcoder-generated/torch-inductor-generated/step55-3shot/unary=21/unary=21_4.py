
import torch.nn as nn

class ModelTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)

    def forward(self, x):
        v1 = torch.tanh(self.conv1(x))
        v2 = torch.tanh(self.conv2(x))
        v3 = torch.tanh(self.conv3(x))
        v4 = torch.tanh(self.conv4(x))
        v5 = torch.tanh(self.conv5(x))
        return v1 + v2 + v3 + v4 + v5
# Inputs to the model
x = torch.randn(1, 1, 224, 224)
