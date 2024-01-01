
class Model(torch.nn.Module):
    def __init__(self, bias=None):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3, bias=bias)
 
    def forward(self, x1, **kwargs):
        return self.linear(x1)

# Initializing the model
m1 = Model()
bias = torch.Tensor(5)
bias.uniform_()
m2 = Model(bias=bias)

# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(3, 5)
