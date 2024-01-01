
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = torch.nn.Parameter(torch.zeros(3, 6, 4, 4))
    def forward(self, x1):
        v1 = self.p0
        v2 = 1 + v1
        v3 = torch.clamp_min(v2, 0.3)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v1 / 28.6
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
