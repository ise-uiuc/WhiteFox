
class Model(torch.nn.Module):
    def __init__(self, linear_dim=64):
        super().__init__()
        self.lin = torch.nn.Linear(linear_dim, 8 * linear_dim)
 
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
