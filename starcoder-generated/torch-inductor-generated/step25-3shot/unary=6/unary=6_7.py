
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 9, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 9, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(9)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = v1 + v2
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v1 * v5
        v7 = v6 / 6
        v8 = v7 + v6
        v9 = v2 * v6
        v10 = self.bn(v9)
        v11 = self.bn(v8)
        return v11
# Inputs to the model
x1 = torch.randn(2, 8, 64, 64)
