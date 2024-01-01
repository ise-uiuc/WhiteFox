
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x1):
        v2 = self.linear(x1)
        v3 = v2 + other
        v4 = relu(v3)
        return v4

# Initializing the model
other = torch.randn(1, 6)
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 3)
