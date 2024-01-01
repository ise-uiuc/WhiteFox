
class FusionConvBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, 3, stride=2, padding=1, dilation=1, groups=2)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, stride=2, padding=1, dilation=1, groups=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = self.conv2(x)
        v5 = v4 > 0
        v6 = v4 * 0.1
        v7 = torch.where(v2, v1, v3)
        return v7
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
