
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randn(5, 5)
        bias = torch.randn(1)
        self.linear = torch.nn.Linear(8, 8, bias=True)
        self.linear.weight = torch.nn.Parameter(weight)
        self.linear.bias = torch.nn.Parameter(bias)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 5)
other = torch.randn(1, 1, 5)
