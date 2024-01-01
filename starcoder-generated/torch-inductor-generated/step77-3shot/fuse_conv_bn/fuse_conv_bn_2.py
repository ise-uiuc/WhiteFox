
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 3)
        self.bn = torch.nn.BatchNorm2d(32, affine=True)
        self.a = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Conv2d(32, 64, 3, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(64, affine=True)
        self.b = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.a(self.bn(self.conv(x)))
        z = self.b(self.bn1(self.conv1(y)))
        return z
# Inputs to the model
x = torch.randn(20, 32, 56, 56)
