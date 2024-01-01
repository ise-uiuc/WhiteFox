
class Model(torch.nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.linear = torch.nn.Linear(3, c)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        return v1 + other

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
__other__ = torch.randn(1, 1)
