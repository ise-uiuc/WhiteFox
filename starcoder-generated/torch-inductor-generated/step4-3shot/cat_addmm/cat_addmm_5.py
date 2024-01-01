
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, dim):
        v1 = torch.addmm(x1, x2, x3)
        v2 = torch.cat([v1], dim)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3, 3, 3)
x3 = torch.randn(3)
dim = 0
