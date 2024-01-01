
class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = torch.nn.Linear(8, n)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
n = 10
m = Model(n)

# Inputs to the model
x1 = torch.randn(1, n)
other = torch.randn(1, n)
