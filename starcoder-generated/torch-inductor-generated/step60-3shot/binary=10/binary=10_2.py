
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
        self.other = other
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.other
        return v2

# Initializing the model
m1 = Model(torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4))
m2 = Model(torch.randint(1, 32, (1, 4)))

# Inputs to the model
x = torch.randn(1, 8)

# Model forwarding pass
__output1__ = m1(x)
__output2__ = m2(x)

