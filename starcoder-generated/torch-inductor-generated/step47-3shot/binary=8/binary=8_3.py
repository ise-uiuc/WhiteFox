
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4, x5):
        a1 = self.conv1(x1)
        a2 = self.conv1(x2)
        a3 = self.conv1(x3)
        a4 = self.conv1(x4)
        a5 = self.conv1(x5)
        a6 = a1*a2*a3*a4*a5
        a7 = a2.mul_(a1)
        a8 = a6.mul(a7)
        a9 = a1.mul(a2).mul(a3).mul(a4).mul(a5)
        return a8.div(a9)
# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)
x2 = torch.randn(4, 3, 64, 64)
x3 = torch.randn(4, 3, 64, 64)
x4 = torch.randn(4, 3, 64, 64)
x5 = torch.randn(4, 3, 64, 64)
