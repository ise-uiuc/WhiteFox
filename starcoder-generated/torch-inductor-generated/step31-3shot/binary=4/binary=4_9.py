
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 8)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 + self.other
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m1 = Model(torch.randn(1))
m2 = Model(torch.randn(5))

# Inputs to the model
x1 = torch.randn(1, 64)
__output_1__ = m1(x1)
__output_2__ = m2(x1)

