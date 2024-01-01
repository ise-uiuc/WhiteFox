
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.conv = torch.nn.Conv2d(1, 1, 2)
    def forward(self, x):
        y = self.bn(x)
        z = self.conv(y)
        return z
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
