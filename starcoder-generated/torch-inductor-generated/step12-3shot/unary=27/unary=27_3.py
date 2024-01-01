
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.conv = torch.nn.Conv2d(64, 128, 2, stride=1, padding=1, groups=32, bias=False)
        self.bn  = torch.nn.BatchNorm2d(128, eps=0.1, momentum=10)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.bn(v3)
        return v4
min = -2
max = -0.7
# Inputs to the model
x1 = torch.randn(1, 64, 50, 50)
