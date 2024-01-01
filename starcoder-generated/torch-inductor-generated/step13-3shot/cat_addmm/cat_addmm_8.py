
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        t1 = torch.addmm(x1, x2, x3)
        t2 = torch.cat([t1], dim=1)
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 64, 16)
x2 = torch.randn(64, 128)
x3 = torch.randn(64, 128)
