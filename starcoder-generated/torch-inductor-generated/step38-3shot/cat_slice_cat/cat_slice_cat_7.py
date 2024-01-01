
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, size):
        t1 = torch.cat([x3, x1])
        t2 = t1[:, :size]
        t3 = torch.cat([t1, t2], dim=1)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1, x2, x3, x4, size = torch.randn(1, 64), torch.randn(1, 32), torch.randn(1, 16), torch.randn(1, 8), 4
