
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 20)
 
    def forward(self, x1, x2=None):
        v1 = self.lin(x1)
        if x2 is not None:
            v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
