
class Model(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.lin = torch.nn.Linear(6, 3)
        self.p1 = p1

    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = v1 - self.p1
        return v2

# Initializing the model
m = Model(torch.empty(3))

# Inputs to the model
x1 = torch.randn(8, 6)
