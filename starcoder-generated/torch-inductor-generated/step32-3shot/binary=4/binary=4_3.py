
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 4)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v1, v2

# Initializing the model
m = Model()

## Inputs to the model
x1 = torch.randn(1, 1)
other = torch.randn(4)

__output__, __output2__ = m(x1, other)

