
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 1, stride=2)
        self.bn = torch.nn.BatchNorm2d(4)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        z = self.bn(y)
        return z
# Inputs to the model
x = torch.randn(1, 1, 8192, 8192)
