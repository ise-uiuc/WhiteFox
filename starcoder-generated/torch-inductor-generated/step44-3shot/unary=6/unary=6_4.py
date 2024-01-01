
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(10)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = -3 + v1
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 3
        v6 = self.bn(v5)
        v7 = self.relu(v6)
        return v7.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
