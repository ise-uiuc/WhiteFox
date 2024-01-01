
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 2, stride=3, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1, x2):
        v1 = x1 * x2
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0
max = 6
# Inputs to the model
x1 = torch.randn(8, 3, 10, 12)
x2 = torch.randn(8, 3, 10, 12)
