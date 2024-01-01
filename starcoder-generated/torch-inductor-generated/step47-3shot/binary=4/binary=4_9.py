
class Model(torch.nn.Module):
    linear : torch.nn.Linear
 
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 7)
        self.other = other
 
    def forward(self, x1):
        return self.linear(x1) + self.other

# Initializing the model
other = torch.nn.Parameter(torch.rand(1, 7))
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 3)
