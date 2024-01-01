
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 16, 5, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 2, 3, padding=2)
        self.conv3 = torch.nn.Conv2d(2, 8, 5, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 16, 3, padding=2)
        self.conv5 = torch.nn.Conv2d(16, 2, 3, padding=1)
        self.conv6 = torch.nn.Conv2d(16, 2, 1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = torch.clamp_min(v6, self.min)
        v8 = torch.clamp_max(v7, self.max)
        return v8
min = 0.5
max = 1.0
# Inputs to the model
x1 = torch.randn(1, 2, 320, 103)
