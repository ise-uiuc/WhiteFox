
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.other = other
 
    def forward(self, x):
        v1 = self.linear(x)
        return torch.nn.functional.relu(v1 + self.other)

# Initializing the model
other = torch.nn.Parameter(torch.zeros(5))
m = Model(other)

# Inputs to the model
x = torch.randn(1, 10)
