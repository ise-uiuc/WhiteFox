
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 20, 3, stride=2, padding=2)
        self.bn = torch.nn.BatchNorm2d(10)
    def forward(self, x1, other=None, other1=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(10, 10, 3, 1, 1)
        if other1 == None:
            other1 = torch.randn(10)
        v1 = v1 + other
        v2 = self.bn(v1)
        v3 = v2 + 1e-05
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
