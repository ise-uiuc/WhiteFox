
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        t1, t2, t3, t4 = torch.split(x1, [2, 2, 2, 4], dim=1)
        return torch.cat([t1, t2, t4, t3], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 5, 5)
