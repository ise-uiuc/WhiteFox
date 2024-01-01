
import sys
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(256, 33, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
        self.bn = torch.nn.BatchNorm2d(33)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 256, 48, 48)
