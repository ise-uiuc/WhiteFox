
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
        self.conv0 = torch.nn.Conv2d(8, 4, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(4, 2, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(2, 1, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv0(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        v7 = self.conv1(v6)
        v8 = torch.clamp_min(v7, self.min)
        v9 = torch.clamp_max(v8, self.max)
        v10 = self.conv2(v9)
        v11 = torch.clamp_min(v10, self.min)
        v12 = torch.clamp_max(v11, self.max)
        return v12
min = 1e-06
max = 100000
# Inputs to the model
x1 = torch.randn(1, 16, 12, 12)
