
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp_max(v2, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.bn(v5)
        v7 = self.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
