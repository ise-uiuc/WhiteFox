
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.other + v1
        return v2

# Initializing the model
__other__ = torch.randn(1, 8)
m = Model(__other__)

# Inputs to the model
x1 = torch.randn(1, 8)
