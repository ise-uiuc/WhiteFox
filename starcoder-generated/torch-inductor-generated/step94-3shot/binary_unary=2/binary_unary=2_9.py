
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x0):
        v1 = self.conv(x0)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = self.bn(v3)
        v5 = v4 - 2
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x0 = torch.randn(1, 3, 32, 32)
