
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=2, padding=3)
        self.bn = torch.nn.BatchNorm2d(32, eps=2.7667575710663685e-05, momentum=0.1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = -150
max = 0
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
