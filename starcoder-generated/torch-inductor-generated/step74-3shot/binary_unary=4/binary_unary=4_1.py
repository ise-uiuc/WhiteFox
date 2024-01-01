
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.other = other
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 + self.other
        x4 = torch.nn.functional.relu(x3)
        return x4

# Initializing the model
other = torch.nn.Parameter(torch.randn(8, 8))
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 8)
