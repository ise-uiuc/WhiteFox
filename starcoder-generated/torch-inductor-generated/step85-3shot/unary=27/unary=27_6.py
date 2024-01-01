
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.conv = torch.nn.Conv2d(96, 96, 3, stride=1, padding=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -29.67620086669922
max = -10.912040710449219
# Inputs to the model
x1 = torch.randn(1, 96, 17, 16)
