
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1 + x1

# Initializing the model
m = Model(16)

# Inputs to the model
x1 = torch.randn(2, 16)
