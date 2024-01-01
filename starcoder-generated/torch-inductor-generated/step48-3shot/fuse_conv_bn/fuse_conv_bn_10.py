
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, groups=2)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x4):
        x4 = self.bn(self.conv(x4))
        return x4
# Inputs to the model
x4 = torch.randn(1, 6, 4, 4)
