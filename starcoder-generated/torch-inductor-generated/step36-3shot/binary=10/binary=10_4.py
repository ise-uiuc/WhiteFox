
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3, 16)
 
    def forward(self, x1, other):
        v1 = self.l1(x1)
        return v1 + other

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
__other__ = torch.randn(1, 16)
