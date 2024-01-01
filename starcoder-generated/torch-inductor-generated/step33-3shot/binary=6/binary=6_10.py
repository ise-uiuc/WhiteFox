
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.utils.prune.PruningContainer()
        self.f.i = torch.nn.Linear(1, 1)
        self.f.i2 = torch.nn.Linear(1, 1)
 
    def forward(self, x1, x2, x3):
        v1 = self.f.i(x1)
        v2 = self.f.i2(x2)
        v3 = v1 - x3
        v4 = v2 - v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
x3 = torch.randn(1, 1)
