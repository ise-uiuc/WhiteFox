
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = self.linear(x1) + self.weight
        return v1

# Initializing the model
m = Model()

# Parameters of the model
m.weight = torch.nn.Parameter(torch.randn(1, 3))

# Inputs to the model
x1 = torch.randn(1, 3)
