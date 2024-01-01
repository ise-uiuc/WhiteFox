
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1, stride=1, padding=0)
        self.conv2d2 = torch.nn.Conv2d(1, 3, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv2d2(v3)
        return v4
min = 1.4
max = 0.8
# Inputs to the model
x1 = torch.randn(1, 2, 224, 224)
