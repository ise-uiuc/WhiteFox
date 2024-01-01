
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8, bias=False)

    def forward(self, x1, x2=0, x3=10):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, x2)
        v3 = torch.clamp_max(v2, x3)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
