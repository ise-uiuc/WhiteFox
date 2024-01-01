
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 9)

    def forward(self, x0):
        v0 = self.linear(x0)
        v1 = v0 * torch.clamp(v0 + 3, 0, 6)
        v2 = v1 / 6
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 6)
