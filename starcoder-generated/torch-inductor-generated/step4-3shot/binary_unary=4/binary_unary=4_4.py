
class Model(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.linear = torch.nn.Linear(64, 1)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(torch.zeros(64, 1))

# Inputs to the model
x1 = torch.randn(64, 64)
other = torch.rand(64, 1)
