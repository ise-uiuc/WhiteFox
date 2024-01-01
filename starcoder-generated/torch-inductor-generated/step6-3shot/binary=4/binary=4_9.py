
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
        self.other = other

    def forward(self, x1):
        return self.linear(x1) + self.other

# Initializing the model
m = Model(torch.randn(1, 8))

# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
