
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16384)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + __other__
        return v2

# Initializing the model
m = Model()

# Inputs to the model
__x1__ = torch.randn(1, 64)
__other__ = torch.randn(1, 16384)
