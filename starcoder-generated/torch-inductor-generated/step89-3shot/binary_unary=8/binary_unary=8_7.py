
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn(v1)
        v3 = self.conv1(x1)
        v4 = self.bn(v3)
        v5 = v2 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
