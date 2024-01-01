
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(24)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.sigmoid(v1)
        v3 = self.bn(v2)
        v4 = self.conv(v3)
        v5 = F.sigmoid(v4)
        v6 = self.conv(v5)
        v7 = F.sigmoid(v6)
        v8 = self.conv(v7)
        v9 = F.sigmoid(v8)
        v10 = self.conv(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
