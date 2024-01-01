
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(12)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -1000
max = 1500
# Inputs to the model
x1 = torch.randn(5, 12, 28, 28)
