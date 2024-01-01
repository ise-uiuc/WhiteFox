
class Model(torch.nn.Module):
    def __init__(self, linear, bias):
        super().__init__()
        self.linear = linear
        self.bias = bias
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.bias
        return v2

# Initializing the model
lin = torch.nn.Linear(64, 64)
b = torch.nn.Parameter(torch.randn(64, 64))
m = Model(lin, b)

# Inputs to the model
x1 = torch.randn(1, 64)
