
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x3):
        x = self.conv1(x3)
        y = self.bn(x)
        z1 = self.conv1(y)
        z2 = self.bn(z1)
        y = self.conv1(z2)
        return y
# Inputs to the model
x5 = torch.randn(1, 3, 4, 4)
