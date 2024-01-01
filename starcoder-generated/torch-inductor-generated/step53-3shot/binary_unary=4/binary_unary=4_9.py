
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other = other
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.other
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model(torch.nn.Parameter(torch.ones(8, 3)))

# Inputs to the model
x = torch.randn(5, 3)
