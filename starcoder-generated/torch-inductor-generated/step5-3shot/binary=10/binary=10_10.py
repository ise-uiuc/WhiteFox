
class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = torch.nn.Linear(n, 1)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model(100)

# Inputs to the model
x1 = torch.randn(10, 100)
