
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 8)
        self.other = torch.nn.Parameter(torch.randn(8))
 
    def forward(self, x1, *, other):
        v1 = self.lin(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 4)
other = torch.randn(8)
