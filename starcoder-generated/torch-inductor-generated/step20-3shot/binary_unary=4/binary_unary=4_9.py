
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x1, other=None):
        if other is None:
            return self.linear(x1)
        else:
            v1 = self.linear(x1)
            v2 = v1 + other
            return torch.nn.functional.relu(v2)

# Initializing the model
m = Model(other=torch.randn(1))

# Inputs to the model
x1 = torch.randn(1)
