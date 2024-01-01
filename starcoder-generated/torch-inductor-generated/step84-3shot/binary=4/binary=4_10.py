
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32, bias=True)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        return v1 + other

# Initializing it
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
__other__ = torch.randn(1, 32)
