
class Model(torch.nn.Module):
    def __init__(self, dim=3, other=3.5):
        super().__init__()
        self.dim = dim
        self.linear = torch.nn.Linear(dim, 1)
        self.other = other
 
    def forward(self, x):
        v2 = self.linear(x) - self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
