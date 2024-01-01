
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = v2 - 2.0
        v4 = F.relu(v3)
        v5 = self.conv(v4)
        v6 = self.bn(v5)
        v7 = v6 - 2.0
        v8 = F.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
