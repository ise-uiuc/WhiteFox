
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
 
    def forward(self, x1, x2, x3):
        t1 = torch.cat([x1, x2, x3], dim=1)
        t2 = t1[:, 0:]
        t3 = t2[:, 0:]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 1024, 7, 7)
x2 = torch.randn(16, 512, 14, 14)
x3 = torch.randn(16, 256, 28, 28)
