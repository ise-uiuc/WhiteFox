
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 51, stride=35, padding=8, dilation=48)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -0.8636441
max = 0.98630943
# Inputs to the model
x1 = torch.randn(1, 3, 67, 167)
