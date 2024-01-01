
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)

    def forward(self, x1, x2=None):
        if x2 is not None:
            v1 = self.linear(x1)
            v2 = v1 + x2
            return v2
        return self.linear(x1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100)
x2 = torch.randn(100)
