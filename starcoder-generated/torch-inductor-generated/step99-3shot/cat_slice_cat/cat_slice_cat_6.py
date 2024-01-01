
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, i1, i2, i3, i4):
        t1 = torch.cat([i1, i2, i3, i4], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:self.size]
        return torch.cat([t1, t3], dim=1)

# Initializing the model
m = Model(size)

# Inputs to the model
size = 64
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)
