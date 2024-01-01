
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.avg_pool(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.0589855
max = -2.01713
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
