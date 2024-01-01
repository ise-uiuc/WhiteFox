
class Model(torch.nn.Module):
    def __init__(self, min, max, min_a, max_a):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 5, 1, stride=1, padding=0)
        self.min = min
        self.max = max
        self.min_a = min_a
        self.max_a = max_a
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v5 = self.conv(x2)
        v6 = torch.clamp_min(v5, self.min)
        v9 = torch.clamp_max(v6, self.max)
        v10 = self.conv(x3)
        v11 = torch.clamp_min(v10, self.min)
        v12 = torch.clamp_max(v11, self.max)
        v8 = x4.add(1)
        v7 = v8.add(1)
        v4 = v7.add(1)
        v14 = v3.mul(v4)
        v15 = v9.mul(v12)
        v16 = v14.add(v15)
        return v16
min = 7
max = 7.6
# Inputs to the model
x1 = torch.randn(64, 4, 33, 32)
x2 = torch.randn(64, 5, 30, 31)
x3 = torch.randn(64, 5, 25, 24)
x4 = torch.randn(64, 5, 10, 13)
