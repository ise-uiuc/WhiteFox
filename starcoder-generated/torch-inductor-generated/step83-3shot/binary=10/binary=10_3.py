
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.other)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model(torch.randn(5, 3))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
__output1__ = m(x1)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
__output2__ = m(x1)

