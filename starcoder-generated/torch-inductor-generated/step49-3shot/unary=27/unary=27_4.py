
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 40, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(40, 80, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(80, 10, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v5 = torch.clamp_max(v2, self.max)
        v6 = self.conv2(v5)
        v7 = torch.clamp_min(v6, self.min)
        v8 = torch.clamp_max(v7, self.max)
        v10 = self.conv3(v8)
        v11 = torch.clamp_min(v10, self.min)
        v12 = torch.clamp_max(v11, self.max)
        return v12
min = 5
max = 1
# Inputs to the model
x1 = torch.randn(1, 10, 30, 30)
