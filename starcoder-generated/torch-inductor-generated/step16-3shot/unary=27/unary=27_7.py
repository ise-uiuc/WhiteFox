
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, stride=2, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        x1 = self.conv(x1)
        x2 = torch.clamp_min(x1, min=self.min)
        x3 = torch.clamp_max(x2, max=self.max)
        return x3
min = 0.0
max = -0.3
# Inputs to the model
input = torch.randn(1, 1, 64, 64)
