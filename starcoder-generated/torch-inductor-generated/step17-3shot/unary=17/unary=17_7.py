
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, padding=0, stride=2)
        self.bn = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.AvgPool2d(3, padding=0, stride=2, ceil_mode=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.pool(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
