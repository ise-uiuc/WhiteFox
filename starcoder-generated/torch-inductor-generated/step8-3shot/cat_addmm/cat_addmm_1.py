
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6):
        res  = x1[:, 0] * x2[:, 1] + x3[:, 0] * x4[:, 1]
        res2 = torch.cat([res], dim=0)
        res3 = res2 + 1.0
        res4 = res3[:, 0] + 1.0
        out = torch.cat([res4], dim=0)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
x3 = torch.randn(1, 2)
x4 = torch.randn(1, 2)
x5 = torch.randn(1, 2)
x6 = torch.randn(1, 2)
