
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(2, 3, 1)
    def forward(self, x1):
        x2 = self.bn(x1)
        x3 = self.bn(x1)
        x1 = self.bn(x2)
        x2 = self.conv1(x1)
        x3 = self.conv1(x2)
        v1 = self.bn(x3)
        v2 = self.conv1(self.bn(x3))
        x4 = self.conv1(x1)
        x6 = torch.add(x3, x1)
        x2 = self.bn(x6)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
