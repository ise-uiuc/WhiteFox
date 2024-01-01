
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.conv(v1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
