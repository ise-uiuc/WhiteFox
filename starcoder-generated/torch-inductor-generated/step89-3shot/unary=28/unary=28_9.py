
class Model(torch.nn.Module):
    def __init__(self, _min, _max):
        super().__init__()
        self.linear = torch.nn.Linear(5, 1)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, _min)
        return torch.clamp_max(v2, _max)

# Initializing the model
_min = 0.1
_max = 0.3
m = Model(_min, _max)

# Input to the model
x1 = torch.randn(1, 5)
x1_min   = torch.min(x1)
x1_max   = torch.max(x1)
