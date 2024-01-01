
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)

    def forward(self, x1, other=torch.ones(1, 8, 1, 1)):
        return self.linear(x1) - other

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 16, 16)
