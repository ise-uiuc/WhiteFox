
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        v3 = v2 - 0.0
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
