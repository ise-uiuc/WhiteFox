
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)
        self.other = other.unsqueeze(0) * 0.5

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the inputs
x2 = torch.rand(1, 3)

# Initializing the model
m = Model(x2)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
