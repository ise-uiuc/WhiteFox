
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        return torch.add(v1, other)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 16)
other = torch.randn(2, 16)
__output1__ = m(x1, other)
__output2__ = m(x1, other)
__output3__ = m(x1, other)
## Input requirements
