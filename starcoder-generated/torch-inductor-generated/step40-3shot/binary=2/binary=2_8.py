
class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v = self.conv(x)
        v = v - 23.4
        return v
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = A()
    def forward(self, x):
        v1 = self.a(x)
        v2 = v1 / 36
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
