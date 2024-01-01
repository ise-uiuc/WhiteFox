
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        x1 = self.conv(x1)
        x2 = self.conv(x1)
        x3 = self.conv(x2)
        x3 = self.conv(x3)
        x4 = self.conv(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
