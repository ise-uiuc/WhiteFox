
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1) + 3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 * v1
        v5 = v4 / 6
        v6 = self.bn(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
