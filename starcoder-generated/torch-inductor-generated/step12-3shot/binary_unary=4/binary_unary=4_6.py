
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = v2.relu()
        return v2

# Initializing the model
other = torch.randn(1, 1, 64, 64)
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
