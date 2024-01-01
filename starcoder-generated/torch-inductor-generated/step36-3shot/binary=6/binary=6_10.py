
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v1

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.nn.Parameter(torch.randn(1, 5))
x1 = torch.randn(1, 5)
