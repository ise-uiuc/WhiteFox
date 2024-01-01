
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10,5, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 1.503006003006
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)

# Initializing the linear layer (this will generate a random weight and bias for the linear transformation)
m.linear = torch.nn.Linear(10, 5, bias=True)
