
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = self.bn(v3)
        v5 = v1 - 0.1
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
