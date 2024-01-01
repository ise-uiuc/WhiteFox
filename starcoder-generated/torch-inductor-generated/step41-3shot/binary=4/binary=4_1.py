
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 128)
 
    def forward(self, x1, other=None):
        if other is None:
            other = torch.nn.Parameter(torch.rand(128,))
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(128)
