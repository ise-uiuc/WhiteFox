
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(64)
    def forward(self, x2, x3):
        v1 = self.conv(x2)
        v1 += x3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 / 6
        v5 = self.bn(v4)
        return v5
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
x2 = torch.randn(3, 3, 64, 64)
