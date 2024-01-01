
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
        self.other = other
 
    def forward(self, x1):
        z = self.linear(x1)
        y = z + self.other
        return y

# Initializing the model
m1 = Model(torch.randn(4, 8))
m2 = Model(torch.randn(4, 8))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 8)
