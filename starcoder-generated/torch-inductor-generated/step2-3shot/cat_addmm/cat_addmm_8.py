
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.addmm(x1, x2, x3)
        v2 = torch.cat([v1], 2)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 112, 112)
x2 = torch.randn(1, 64, 112, 112)
x3 = torch.randn(64, 1000)
