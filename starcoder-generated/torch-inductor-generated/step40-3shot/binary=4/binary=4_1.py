
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10, bias=False)

    def forward(self, x1, other):
        v1 = self.linear(x1 + other)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 1, 1)
other = torch.randn(1, 10, 1, 1)
