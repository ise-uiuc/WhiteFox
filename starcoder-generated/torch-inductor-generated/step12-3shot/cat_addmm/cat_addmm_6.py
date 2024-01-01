
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.addmm(x1.mean(), x2, x3)
        v2 = torch.cat([v1, x4, x5], dim=0)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(4, 1)
x3 = torch.randn(4, 1)
x4 = torch.randn(1, 1)
x5 = torch.randn(1, 1)
