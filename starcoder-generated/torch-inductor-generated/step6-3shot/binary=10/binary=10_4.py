
class Model(torch.nn.Module):
    def __init__(self, o):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8, bias=True)
        self.other = o
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
other = torch.nn.Parameter(torch.randn(8, 32))
m = Model(other)

# Inputs to the model
x1 = torch.randn(10, 32)
