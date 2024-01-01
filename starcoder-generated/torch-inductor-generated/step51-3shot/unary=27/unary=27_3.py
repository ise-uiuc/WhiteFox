
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 3.0
max = 4.0
# Inputs to the model
x1 = torch.randn(4, 3)
