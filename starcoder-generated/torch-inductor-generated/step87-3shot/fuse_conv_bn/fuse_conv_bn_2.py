
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, groups=3)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = self.bn(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
