
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 64)
        self.linear2 = torch.nn.Linear(64, 2)

    def forward(self, x1, other):
        v1 = self.linear1(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16)
x2 = torch.randn(64)
