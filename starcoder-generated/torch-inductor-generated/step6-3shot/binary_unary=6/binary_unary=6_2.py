
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(32, 32)
        self.m2 = torch.nn.Linear(32, 32)
 
    def forward(self, x1, other, **kw):
        v1 = self.m1(x1)
        v2 = torch.sub(v1, other)
        v3 = torch.clamp(v2, min=0.0)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 32)
other = torch.randn(1, 32)
