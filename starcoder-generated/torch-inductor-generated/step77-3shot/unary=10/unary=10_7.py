
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 9)

    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = v5 * x2
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 9)
