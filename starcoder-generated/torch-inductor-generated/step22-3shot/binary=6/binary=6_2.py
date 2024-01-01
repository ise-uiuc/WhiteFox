
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 8)
 
    def forward(self, x1):
        v2 = x1 - other
        return self.linear(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(10, 1)
v1 = other ** 2
