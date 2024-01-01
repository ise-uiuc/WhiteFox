
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(4, stride=4, padding=3, ceil_mode=False)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0
max = 0.1
# Inputs to the model
x1 = torch.randn(1, 5, 24, 24)
