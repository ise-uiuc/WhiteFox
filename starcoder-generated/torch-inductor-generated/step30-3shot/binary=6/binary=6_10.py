
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(1000, 1000)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        return v2

# Initializing other
other = torch.tensor([0.0361])

# Initializing the model
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 1000)
