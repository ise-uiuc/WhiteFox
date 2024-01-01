
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 64, 1, 1)
m.linear.weight = torch.nn.Parameter(torch.ones_like(m.linear.weight))
__other__ = torch.randint(2, size=(1, 64, 1, 1))
