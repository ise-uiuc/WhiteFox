
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, 1)
 
    def forward(self, x1, x2, other):
        v1 = self.linear(x1)
        v2 = v1 + x2 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2, 3, 3)
