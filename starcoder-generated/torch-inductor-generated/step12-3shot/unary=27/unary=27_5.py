
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 4, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = torch.mean(x1)
        v2 = v1.expand(x1.size())
        v3 = self.conv(v2)
        v4 = torch.clamp_min(v3 + v2, self.min)
        v5 = torch.clamp_max(v4 + v3, self.max)
        return v5
min = -0.8
max = 0.7
# Inputs to the model
x1 = torch.randn(1, 8, 100, 100)
