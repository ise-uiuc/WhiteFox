
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 2, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 2, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        v5 = v4.permute(0, 2, 3, 1).view(100, 100, -1)
        return v5.mean(0)
min = -1.5
max = 0
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
