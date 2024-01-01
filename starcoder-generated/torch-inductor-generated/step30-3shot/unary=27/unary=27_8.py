
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(16, 8, 5, stride=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.maxpool(x1)
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = -0.5
max = 0.7
# Inputs to the model
x1 = torch.randn(1, 16, 100, 110)
