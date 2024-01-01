
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(40, 10, 5, stride=1, padding=0, groups=4)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v5 = torch.clamp_max(v2, self.max)
        return v5
min = 5
max = 0.2
# Inputs to the model
x1 = torch.randn(1, 40, 21, 90)
