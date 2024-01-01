
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(12, 6)
        self.other = other
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.other
        return v2

# Initializing the model
other = torch.randn(6)
m = Model(other)

# Inputs to the model
x = torch.randn(1, 12)
