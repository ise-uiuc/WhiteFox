
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 20, 5, stride=3, padding=6)
        self.bn = torch.nn.BatchNorm2d(20)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        return self.bn(v5)
# Inputs to the model
x1 = torch.randn(1, 20, 28, 28)
