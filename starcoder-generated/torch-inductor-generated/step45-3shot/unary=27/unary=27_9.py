
class Model(torch.nn.Module):
    def __init__(self, max, min):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.max = max
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
max = 0.001
min = 0.815
# Inputs to the model
x1 = torch.randn(1, 3, 200, 200)
