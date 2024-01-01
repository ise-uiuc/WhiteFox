
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False)
        self.other = Parameter(torch.randn(16, 1, 4, 4))

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 5, 5)
