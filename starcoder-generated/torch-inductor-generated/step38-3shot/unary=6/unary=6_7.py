
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 100, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(100)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.bn(v5)
        v7 = self.tanh(v6)
        return v7
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
