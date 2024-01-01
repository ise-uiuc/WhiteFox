
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        t1 = torch.cat([x1, x1])
        t2 = t1[:, 32:64]
        t3 = t2[:, 16:32]
        return torch.cat([t1, t3], dim=1)

# Initializing model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 24, 24)
