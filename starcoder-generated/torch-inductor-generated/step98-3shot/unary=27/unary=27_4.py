
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 7, 3, stride=1, padding=1)
        self.conv_p = torch.nn.Conv2d(7, 4, 3, stride=1, padding=1)
        self.conv_pp = torch.nn.Conv2d(4, 6, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv_p(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        v7 = self.conv_pp(v6)
        v8 = torch.clamp_min(v7, self.min)
        v9 = torch.clamp_max(v8, self.max)
        return v9
min = 0.3
max = 0.81
# Inputs to the model
x1 = torch.randn(1, 10, 234, 987)
x2 = torch.randn(1, 10, 234, 987)
