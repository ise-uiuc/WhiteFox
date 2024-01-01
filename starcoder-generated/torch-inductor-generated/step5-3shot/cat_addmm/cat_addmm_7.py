
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, m1, m2):
        v1 = torch.addmm(x1, m1, m2)
        v2 = torch.cat([v1], dim)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
m1 = torch.randn(8, 64)
m2 = torch.randn(8, 64)
