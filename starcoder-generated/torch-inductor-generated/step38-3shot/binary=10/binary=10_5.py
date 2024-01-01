
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
 
    def forward(self, x1, __other__):
        v1 = self.linear(x1)
        v2 = v1 + __other__
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
__other__ = torch.randn(1, 3)
