
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(dim, 8)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
dim = 10 # Other tensor dimension
other = torch.randn(dim)
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, dim)
