
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 56, stride=1, padding=16)
        self.bn = torch.nn.BatchNorm2d(6)
    def forward(self, x1):
        e1 = self.conv(x1)
        e2 = e1 + 3
        e3 = torch.clamp_min(e2, 0)
        e4 = torch.clamp_max(e3, 6)
        e5 = e1 * e4
        e6 = e5 / 6
        e7 = self.bn(e6)
        return e7
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
