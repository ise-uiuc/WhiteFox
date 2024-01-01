
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1, __other__=None):
        v1 = self.linear(x1)
        v2 = v1 + __other__
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(4, 3))

# Inputs to the model
x1 = torch.randn(1, 3)
__other__ = torch.randn(4, 3)
