
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 4, 1, stride=1, bias=False)
    def forward(self, x1, x2=torch.randn((1, 2, 16, 16)), x3=1, x4=5, padding1=None, padding2=None, padding3=None, padding4=None, x5=None):
        x1 = self.conv(x1)
        v1 = x4
        v3 = self.conv(x3)
        v4 = x1
        v2 = v3 + v4
        return v2
# Inputs to the model
x1 = 1
x2 = 1
x3 = torch.randn(1, 3, 16, 16)
x4 = torch.randn(1, 4, 16, 16)
x5 = torch.zeros((1, 3, 16, 16))
