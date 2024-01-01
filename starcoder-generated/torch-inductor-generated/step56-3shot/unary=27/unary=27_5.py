
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=8, padding=0, dilation=7)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -0.001
max = 0
# Inputs to the model
x1 = torch.randn(1, 1, 100, 1)
