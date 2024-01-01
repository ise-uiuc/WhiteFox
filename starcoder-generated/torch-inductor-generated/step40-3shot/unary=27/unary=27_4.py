
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = torch.clamp_min(x1, self.min)
        v2 = torch.clamp_max(v1, self.max)
        return v2
min = 0.0034704288624572754
max = 0.8876201820373535
# Inputs to the model
x1 = torch.randn(2, 3, 2, 2)
