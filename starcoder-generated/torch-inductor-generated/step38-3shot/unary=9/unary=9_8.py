
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.nn.Parameter(28.0 * torch.ones(1))
    def forward(self, x2):
        v1 = self.v1 + x2
        v2 = v1 + x2
        v3 = v2 + 3
        v4 = v3.clamp(min=0, max=6)
        v5 = v4 / 6
        v6 = v5 * 4
        v7 = v3 + 5
        v8 = v7.clamp(min=0, max=6)
        v9 = v8 / 6
        return v6 + v9
# Inputs to the model
x2 = torch.randn(1, 1, 1, 1)
