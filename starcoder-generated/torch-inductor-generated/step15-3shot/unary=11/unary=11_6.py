
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(23, 2)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.linear(v1)
        v3 = torch.cat([v1, v2], dim=1)
        v4 = v3 + 3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 23)
