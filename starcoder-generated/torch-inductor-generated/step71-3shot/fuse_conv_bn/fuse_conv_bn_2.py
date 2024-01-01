
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, groups=1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y.reshape(3, -1))
        return y
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
