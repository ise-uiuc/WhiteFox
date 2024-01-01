
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(8, affine=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = v2.clamp(0, 6)
        v4 = v3.div(6)
        v5 = self.bn(v4)
        v6 = self.other_conv(v5)
        v7 = 3 + v6
        v8 = v7.clamp(min=0, max=6)
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
