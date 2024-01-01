
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(5, 5, 11, stride=1, padding=5, groups=3)
        self.conv1 = torch.nn.Conv2d(5, 30, 2, stride=1, padding=1, groups=3)
        self.conv2 = torch.nn.Conv2d(30, 8, 1, stride=1, padding=0, groups=3)
        self.max = max
        self.min = min
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv1(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        v7 = self.conv2(v6)
        v8 = torch.clamp_min(v7, self.min)
        v9 = torch.clamp_max(v8, self.max)
        return v9
min = 2.7
max = -0.2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
