
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.lin = torch.nn.Linear(100, 200)
 
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(1, 100))

# Inputs to the model
x1 = torch.randn(1, 100)
