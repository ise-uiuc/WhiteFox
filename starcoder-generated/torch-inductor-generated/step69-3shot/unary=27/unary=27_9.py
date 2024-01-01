
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.linear = torch.nn.Linear(1000, 56)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -1.63
max = 0.005
# Inputs to the model
x1 = torch.randn(100, 1000)
