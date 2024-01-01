
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m1 = Model(torch.randn(1, 100, 1, 1))
m2 = Model(torch.randn(1, 100, 1, 1)
# Inputs to the model
x1 = torch.randn(1, 100, 1, 1)
x2 = torch.randn(1, 100, 1, 1)
v1 = m1(x1)
v2 = m2(x2)

