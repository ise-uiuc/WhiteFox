
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 14, stride=1, padding=7)
        self.conv2 = torch.nn.Conv2d(3, 1, 33, stride=2, padding=16)
        self.conv3 = torch.nn.Conv2d(3, 1, 53, stride=1, padding=26)
        self.min = min
        self.max = max
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        r1 = torch.clamp_min(v1, self.min)
        r2 = torch.clamp_min(v2, self.min)
        r3 = torch.clamp_min(v3, self.min)
        r4 = torch.clamp_max(r1, self.max)
        r5 = torch.clamp_max(r2, self.max)
        r6 = torch.clamp_max(r3, self.max)
        v4 = r4 + r5
        v5 = r6 + v4
        return v5
min = -0.33
max = 0.98
# Inputs to the model
x1 = torch.randn(1, 3, 32, 256)
x2 = torch.randn(1, 3, 128, 32)
x3 = torch.randn(1, 3, 64, 64)
