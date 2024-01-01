
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = torch.softmax(v1, 0)
        v3 = v2 + 1.
        v4 = torch.sigmoid(v3)
        return v4
x1 = torch.randn(3, 3, 107, 107)
