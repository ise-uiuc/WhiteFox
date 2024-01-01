
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        y1, y2 = torch.chunk(x1, chunks=2, dim=0)
        z1, z2, z3 = torch.chunk(y1, chunks=3, dim=1)
        return z1, z2, z3

# Initializing the model 
m = Model()

# Inputs to the model
x1 = torch.randn(6, 57, 224, 224)
y1, y2, y3 = m(x1)
