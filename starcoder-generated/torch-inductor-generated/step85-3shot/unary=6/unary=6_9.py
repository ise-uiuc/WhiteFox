
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 95, 3, stride=1, padding=0, groups=95, dilation=1)
        self.bn = torch.nn.BatchNorm2d(95 * 3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.bn(v6)
        v8 = v7.reshape((-1, 3, 1, 95))
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
