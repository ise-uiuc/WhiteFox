
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 20)
 
    def forward(self, x1, x2=None, other=None):
        v1 = self.linear(x1)
        if x2 is not None:
            v2 = v1 + x2
        if other is not None:
            v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
x2 = torch.randn(1, 20)
other = torch.randn(1, 20)
