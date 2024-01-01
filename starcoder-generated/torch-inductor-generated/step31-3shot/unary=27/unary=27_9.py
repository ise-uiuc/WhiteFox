
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 8, 5, groups=3, padding=1)
        self.conv1 = torch.nn.Conv2d(1, 8, 1, groups=2)
        self.avg = torch.nn.AvgPool2d(3, stride=2, padding=0)
        self.max = torch.nn.MaxPool2d((2, 2), stride=(1, 2))
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_max(v1, self.max)
        v3 = torch.clamp_max(v2, min=self.max)
        v4 = self.conv1(v3)
        v5 = torch.clamp_min(v4, self.max)
        v6 = torch.clamp_min(v5, min=self.min)
        v7 = self.avg(v6)
        v8 = torch.clamp_min(v7, max=self.min)
        v9 = torch.clamp_min(v8, min=self.min)
        v10 = torch.clamp_max(v9, max=self.max)
        v11 = self.max(v10)
        v12 = torch.clamp_min(v11, self.max)
        v13 = torch.clamp_max(v12, self.min)
        return v13
min = -0.1
max = 0.05
# Inputs to the model
x1 = torch.randn(1, 20, 98, 97)
