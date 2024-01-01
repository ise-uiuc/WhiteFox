
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)
        self.other = torch.nn.Parameter(other.data)
 
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model with tensors of different sizes
other = torch.randn(5, 3)  # The size needs to match the output of the linear transformation
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 3)
