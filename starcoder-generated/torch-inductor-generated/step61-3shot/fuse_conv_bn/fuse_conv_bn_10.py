
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        v = self.conv(x)
        v1 = self.conv1(v)
        v2 = self.bn(v)
        v3 = self.conv1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
