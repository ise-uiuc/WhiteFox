
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other=None):
        if other is None:
            v1 = self.linear(x1)
        else:
            v1 = self.linear(x1) + other
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(2, 64, 64)
