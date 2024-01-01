
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.linear = torch.nn.Linear(5, 5)
    def forward(self, x1):
        v1 = self.linear(x1)
        v1 = torch.clamp_min(v1, 0.0)
        v1 = torch.clamp_max(v1, self.max)
        return v1
min = 3
max = 1.0
# Inputs to the model
x1 = torch.randn(1, 5)
