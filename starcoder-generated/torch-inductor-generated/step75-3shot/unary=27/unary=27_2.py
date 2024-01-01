
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = torch.clamp_min(x1, self.min)
        v2 = self.conv(v1)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.0
max = 0.75
# Inputs to the model
x1 = torch.randn(1, 2, 10, 10)
