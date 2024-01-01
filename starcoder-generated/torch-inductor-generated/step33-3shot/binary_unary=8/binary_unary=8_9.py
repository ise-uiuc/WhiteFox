
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 32, 3, stride=2, padding=0)
        self.bn = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.bn(v1 + v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
