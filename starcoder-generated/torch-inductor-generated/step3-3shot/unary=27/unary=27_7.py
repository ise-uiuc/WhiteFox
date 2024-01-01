
class Model(torch.nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 2, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = torch.max(v3, x2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
