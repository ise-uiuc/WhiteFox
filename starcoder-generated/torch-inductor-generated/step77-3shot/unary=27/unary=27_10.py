
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.avg = torch.nn.AdaptiveAvgPool2d((10000, 10000))
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.avg(x1)
        v2 = v1.view(-1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 1e-05
max = -1e-05
# Inputs to the model
x1 = torch.randn(1, 100, 100)
