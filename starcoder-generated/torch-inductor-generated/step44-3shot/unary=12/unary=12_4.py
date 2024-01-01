
class ResBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v3 = self.relu(self.bn(x1))
        v2 = self.bn(self.conv(v3))
        v1 = self.bn(self.conv(x1) + v2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
