
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 7, 3)
        self.bn = torch.nn.BatchNorm2d(7)
        self.pool = torch.nn.AvgPool2d(3)
    def forward(self, x8):
        v8 = self.pool(self.conv(x8))
        a8 = self.bn(v8)
        return a8
# Inputs to the model
x8 = torch.randn(6, 7, 56, 56)
