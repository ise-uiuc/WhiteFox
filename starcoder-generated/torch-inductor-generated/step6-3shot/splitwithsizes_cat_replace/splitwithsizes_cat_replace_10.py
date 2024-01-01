
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x0):
        t0 = torch.split(x0, [2, 2, 2, 2, 2, 2, 2, 2], 2)
        c = torch.cat([t1, t2, t3, t4, t5, t6, t7, t8], 2)
        return c[3]

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 8, 64, 64)
