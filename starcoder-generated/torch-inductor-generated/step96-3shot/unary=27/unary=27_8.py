
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 8, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -0.6373
max = 1.3302
# Inputs to the model
x = torch.randn(1, 1, 3, 7)
