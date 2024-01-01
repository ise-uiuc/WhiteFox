
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(12, 1)
        self.other = other
 
    def forward(self, x1)
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m1 = Model(x2)
m2 = Model(x3)

# Inputs to the model
x1 = torch.randn(1, 12)
x2 = torch.randn(1, 1)
x3 = torch.randn(1, 1)
v1 = m1(x1)
v2 = m1(x2)
v3 = m2(x3)

