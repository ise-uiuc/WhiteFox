
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.bn(v6)
        v8 = (v7 + v6).pow(2)
        v9 = v8.sum(axis=2)
        v10 = v9.sqrt()
        v11 = v10 + 3
        v12 = v11.ceil()
        v13 = v12.int()
        v14 = v12.bool()
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
