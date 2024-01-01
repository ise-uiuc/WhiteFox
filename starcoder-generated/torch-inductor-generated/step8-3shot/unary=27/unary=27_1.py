
class Model(torch.nn.Module):
    def __init__(self, min=0.5, max=-0.5):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, stride=2, padding=7)
        self.min = min
        self.max = max
    def forward(self, x1, x2):
        v1 = self.conv(x2)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min =
# Inputs to the model
x1 = torch.randn(1, 3, 110, 110)
x2 = torch.randn(1, 3, 111, 112)
