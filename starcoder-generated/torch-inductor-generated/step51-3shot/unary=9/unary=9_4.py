
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = v2.relu()
        v4 = v3.max(6)
        v5 = v4 / 6
        v6 = self.bn(v5)
        v7 = self.other_conv(v6)
        v8 = 3 + v7
        v9, _ = v8.max(6)
        v10 = v9 / 6
        return v10
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
