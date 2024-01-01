
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v2 = self.conv(v2)
        v2 = self.bn(v2)
        v2 = v2 + self.conv(self.conv(v2/2))
        v2 = torch.relu(v2 / 2)
        return self.conv(v2)
x1 = torch.randn(1, 3, 3, 3)
