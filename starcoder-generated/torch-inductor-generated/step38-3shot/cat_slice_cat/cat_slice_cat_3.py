
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x):
        y1 = torch.cat([x[0], x[1], x[2]], dim=1)
        x0 = y1.narrow(1, 0, 9223372036854775807)
        c1 = x0.narrow(1, 0, self.size)
        c2 = x0.narrow(1, 0, self.size//2)
        c3 = x0.narrow(1, 1024, self.size)
        z0 = torch.cat([y1, c1, c2, c3], dim=1)
        return z0

# Initializing the model
m = Model(size=1024)

# Inputs to the model
x1 = torch.randn(1, 3, 2048, 2048)
x2 = torch.randn(1, 3, 2048, 2048)
x3 = torch.randn(1, 3, 2048, 2048)
