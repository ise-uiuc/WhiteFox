
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 1)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        x2 = v1 + self.other
        return x2

# Initializing the model
other = torch.empty(1, 1, dtype=torch.float)
other.data.uniform_(-1.0, 1.0)
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 8)
