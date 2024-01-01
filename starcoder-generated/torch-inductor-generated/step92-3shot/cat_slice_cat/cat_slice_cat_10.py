 
class Model(torch.nn.Module):
    def __init__(self, s_x1):
        super().__init__()
        self.size = s_x1

    def forward(self, *args, **kwds):
        t1 = torch.cat(*args, **kwds)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:self.size]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
size = 30
m = Model(size)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 32, 32)
