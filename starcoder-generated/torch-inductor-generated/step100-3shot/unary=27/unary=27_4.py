
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 120, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(120)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.9
max = -0.7
# Inputs to the model
x1 = torch.randn(1, 4, 30, 30)
