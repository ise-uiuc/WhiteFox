
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1 + self.other

# Initializing the model
m = Model(torch.ones(5))

# Inputs to the model
x1 = torch.randn(1, 3)
