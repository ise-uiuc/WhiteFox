
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(253, 188, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(188, 149, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(149, 12, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv2(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        v7 = self.conv3(v6)
        v8 = torch.clamp_min(v7, self.min)
        v9 = torch.clamp_max(v8, self.max)
        return v9
min = -1.03
max = 1.47
# Inputs to the model
x1 = torch.randn(1, 253, 10, 10)
