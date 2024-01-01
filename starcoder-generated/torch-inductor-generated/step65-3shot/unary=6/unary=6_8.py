
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = v2.clamp(0, 6)
        v4 = v2 * v3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
