...
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.max = max
        self.conv = torch.nn.Conv2d(1, 7, 1, stride=1, padding=2)
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -5.3
max = -2.2
# Inputs to the model
x1 = torch.randn(4, 1, 51, 51)
