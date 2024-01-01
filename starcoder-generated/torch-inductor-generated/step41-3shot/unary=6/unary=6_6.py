
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(3, 1, 5, stride=1, padding=2, groups=3)
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        t2 = self.depthwise(x1)
        t3 = self.conv(t2)
        t4 = 3 + t3
        t5 = torch.clamp(t4, 0, 6)
        t6 = t2 * t5
        t7 = t6 / 6
        t8 = self.bn(t7)
        return t8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
