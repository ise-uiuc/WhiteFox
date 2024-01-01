
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
        self.bias = torch.nn.Parameter(torch.randn(1, 8))
 
    def forward(self, x1, other):
        v0 = self.linear(x1)
        v0 = v0 + other
        return v0
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
other = torch.randn(1, 8)
