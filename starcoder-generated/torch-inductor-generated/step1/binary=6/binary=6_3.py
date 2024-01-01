
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x, other):
        v = self.linear(x)
        v1 = v - other
        return v1

# Initializing and running the model
m = Model()

# Inputs for the model
x = torch.randn(3, 3)
other = torch.randn(3, 5)
out = m(x, other)
