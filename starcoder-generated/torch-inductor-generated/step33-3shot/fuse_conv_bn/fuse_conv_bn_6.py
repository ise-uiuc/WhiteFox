
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.pool = torch.nn.AvgPool2d(3)
    def forward(self, x4):
        y2 = self.conv(x4)
        y3 = self.pool(y2)
        y4 = self.bn(y3)
        y5 = torch.abs(self.conv(y4))
        return y5
# Inputs to the model
x4 = torch.randn(1, 5, 3, 3)
