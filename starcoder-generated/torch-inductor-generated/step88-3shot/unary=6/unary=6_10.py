
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=2, padding=4)
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.conv(v1)
        v3 = 3 + v2
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        if True:
            v6 = v2 * v5
        else:
            v6 = v1 * v5
        return v6 / 6
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
