
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64)
        self.other = other
        
    def forward(self, x0):
        v1 = self.linear(x0)
        v2 = v1 + self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
other = torch.randn(64)
m = Model(other)

# Inputs to the model
x0 = torch.randn(1, 32)
