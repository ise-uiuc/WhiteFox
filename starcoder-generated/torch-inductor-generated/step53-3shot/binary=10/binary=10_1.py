
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1 + other

# Initializing the model
m = Model(other)

# Inputs to the model
other = torch.randn(1, 100)
x1 = torch.randn(1, 100)
