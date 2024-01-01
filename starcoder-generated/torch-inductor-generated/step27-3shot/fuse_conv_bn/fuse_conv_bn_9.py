
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x4):
        v = self.conv2(x4)
        v3 = self.bn(v)
        v4 = torch.abs(self.conv(v3))
        return v4
# Inputs to the model
x4 = torch.randn(1, 3, 3, 3)
