
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 8, 4, stride=2, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        return v2
min = -0.8
max = 0.0
# Inputs to the model
x1 = torch.randn(1, 6, 250, 250)
