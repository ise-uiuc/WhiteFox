 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1, other):
        x2 = self.linear(x1)
        v1 = x2 + other
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8)
other = torch.randn(2, 16)
_OUTPUT = m(x1, other)

