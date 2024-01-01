
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x2, x3=0.5):
        v1 = self.linear(x2)
        v3 = v1 + x3
        v2 = torch.relu(v3)
        return v2

# Initializing the model
m = Model(other)

# Initializing the model with new argument values
m = Model(other, x3=2)

# Inputs to the model
x2 = torch.randn(1, 3, 5, 5)
