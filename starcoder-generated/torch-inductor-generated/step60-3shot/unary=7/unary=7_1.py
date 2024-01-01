
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.clamp(min=-1, max=+6)
        v3 = v2.add(3.0)
        v4 = v3.mul(6.0)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
