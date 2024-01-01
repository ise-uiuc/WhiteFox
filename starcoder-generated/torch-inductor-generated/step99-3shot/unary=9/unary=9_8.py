
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(8, affine=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        x2 = v1 + 3
        t1 = x2.clamp(0, 6)
        t2 = t1.div(6)
        x3 = self.bn(t2)
        v4 = self.other_conv(x3)
        v5 = v4 + 3
        v6 = v5.clamp(min=0, max=6)
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
