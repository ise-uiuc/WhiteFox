
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

m = Model(dim=5)

# Inputs to the model
x = torch.randn(1, 5)

# Specified "other" tensor
other = torch.randn(1, 5)
