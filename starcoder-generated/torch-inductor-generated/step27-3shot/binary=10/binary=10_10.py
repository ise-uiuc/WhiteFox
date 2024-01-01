
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8)

    def forward(self, x1, other):
        x = self.linear(x1)
        return torch.nn.functional.linear(x, other)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2)
