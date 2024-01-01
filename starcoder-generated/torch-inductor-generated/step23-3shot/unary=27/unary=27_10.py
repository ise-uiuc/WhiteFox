
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 10, stride=3, padding=6)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_v)
        v3 = torch.clamp_max(v2, self.max_v)
        return v3
min = 0.9
max = 0.8
# Inputs to the model
x1 = torch.randn(1, 3, 42, 49)
