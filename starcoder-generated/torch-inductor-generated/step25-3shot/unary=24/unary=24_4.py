
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(25, 25, 3, stride=2, padding=0)
        self.bn = torch.nn.BatchNorm2d(25)
    def forward(self, x):
        negative_slope = -0.1
        v1 = self.conv(x)
        v2 = self.bn(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 25, 32, 32)
