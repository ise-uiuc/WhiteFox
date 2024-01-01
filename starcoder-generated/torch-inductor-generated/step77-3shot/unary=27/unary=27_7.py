
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.bmm = torch.nn.Bilinear(1, 1, 3)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.bmm(x1, x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = None
max = 0.08372771448135376
# Inputs to the model
x1 = torch.randn(32, 1, 1)
