
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 12, 3, stride=1, padding=2)
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, -5.0)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 1.3
max = -5.0
# Inputs to the model
x1 = torch.randn(1, 10, 200, 200)
