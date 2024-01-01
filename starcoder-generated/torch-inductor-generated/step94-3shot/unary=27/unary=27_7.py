
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(13, 19, 2, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(19, 17, 1, stride=1, padding=1)
        self.conv_ = torch.nn.Conv2d(1, 3, 1, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv1(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        v7 = self.conv_(x1)
        v8 = torch.clamp_min(v7, self.min)
        v9 = torch.clamp_max(v8, self.max)
        v10 = self.conv(v9)
        return v10
min = 1
max = 1
# Inputs to the model
x1 = torch.randn(1, 13, 128, 128)
