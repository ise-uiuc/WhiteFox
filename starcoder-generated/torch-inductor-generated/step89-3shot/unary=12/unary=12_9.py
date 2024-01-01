
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 12, 5, stride=2, padding=2, dilation=2, groups=3)
        self.bn = torch.nn.BatchNorm2d(12)
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.leakyrelu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
