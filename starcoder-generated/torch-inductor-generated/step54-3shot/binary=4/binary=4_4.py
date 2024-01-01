
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3, bias=False)
        self.other = torch.nn.Parameter(other)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model(x2)

# Inputs to the model
x3 = torch.randn(1, 3)
