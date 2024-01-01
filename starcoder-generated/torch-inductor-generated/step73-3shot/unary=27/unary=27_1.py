
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.min = min
        self.max = max
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
        v10 = self.conv3(v9)
        v11 = torch.clamp_min(v10, self.min)
        v12 = torch.clamp_max(v11, self.max)
        v13 = self.conv4(v12)
        v14 = torch.clamp_min(v13, self.min)
        v15 = torch.clamp_max(v14, self.max)
        return v15
min = 0.1
max = 20.6
# Inputs to the model
x1 = torch.randn(3, 16, 16, 16)
