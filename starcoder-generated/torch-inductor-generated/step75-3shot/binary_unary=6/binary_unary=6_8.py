
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other = nn.Parameter(other.view(8))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the parameters 'other'
other = torch.rand(8)

# Initializing the model
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 3)
