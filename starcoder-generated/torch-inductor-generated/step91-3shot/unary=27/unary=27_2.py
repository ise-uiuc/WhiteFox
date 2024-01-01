
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.add = torch.nn.ModuleList([torch.nn.Conv2d(1, 1, 1), torch.nn.Conv2d(1, 1, 1)])
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.add[0](x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = v3 + v3
        v5 = self.add[1](v4)
        v6 = torch.clamp_min(v5, self.min)
        v7 = torch.clamp_max(v6, self.max)
        return v7
min = 0.5
max = 1.0
# Inputs to the model
x1 = torch.randn(1, 1, 768, 1)
