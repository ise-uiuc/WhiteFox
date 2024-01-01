
class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(16)
        self.conv = torch.nn.Conv2d(16, 32, 2, 1)
    def forward(self, x8):
        x8 = self.bn(x8)
        x8 = self.conv(x8)
        return x8
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 16, 1, 2)
        self.mod = A()
    def forward(self, x6):
        x6 = self.conv(x6)
        x6 = self.mod(x6)
        return x6
# Inputs to the model
x6 = torch.randn(1, 8, 4, 4)
