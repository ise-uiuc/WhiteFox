
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x1, other=None):
        if other is not None:
            v1 = self.linear(x1)
            v2 = v1 + other
            return v2
        else:
            return self.linear(x1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
x2 = torch.randn(1, 100)
