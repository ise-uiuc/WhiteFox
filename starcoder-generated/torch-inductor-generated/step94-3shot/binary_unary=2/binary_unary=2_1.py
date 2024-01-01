
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = v2 - 5
        v4 = F.relu(v3)
        v5 = v4 - 4
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
