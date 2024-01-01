
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 2)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.bn(x1)
        x3 = self.conv(x2)
        x4 = self.bn(x3)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
