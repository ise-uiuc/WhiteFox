
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(24)
        self.conv1 = torch.nn.Conv2d(24, 24, 3, stride=1, padding=1, groups=1)
    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = self.conv1(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7
# Inputs to the model
x1 = torch.randn(2, 24, 128, 128)
