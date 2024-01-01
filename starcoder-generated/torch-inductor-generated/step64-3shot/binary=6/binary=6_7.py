
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 8)
        self.other = other

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - self.other
        return v2

# Initializing the model
m = Model(torch.randn(1, 8))

# Inputs to the model
x1 = torch.randn(1, 8)
