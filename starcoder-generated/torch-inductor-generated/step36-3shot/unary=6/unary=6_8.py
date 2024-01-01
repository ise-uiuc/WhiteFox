
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1, groups=3)
        self.bn = torch.nn.BatchNorm2d(6)
    def forward(self, x1):
        v1 = torch.ceil(self.conv(x1))
        v2 = torch.exp(v1 * 3)
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.bn(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
