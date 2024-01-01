
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 2, 3, stride=1, padding=0)
        self.conv_i = torch.nn.Conv2d(4, 2, 3, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv_i(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        return v6
min = 0.04
max = -0.85
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
x2 = torch.randn(1, 4, 4, 4)
