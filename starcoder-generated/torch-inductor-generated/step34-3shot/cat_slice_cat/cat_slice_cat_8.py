
class Model(torch.nn.Module):
    def __init__(self, size=256):
        super().__init__()
 
    def forward(self, *x):
        t1 = torch.cat(x, dim=1)
        tmp = int(t1.shape[1])
        t2 = t1[:, 0:tmp]
        tmp2 = int(t2.shape[1])
        st1 = t2[:, 0:self.size]
        t3 = torch.cat([t1, st1], dim=1)
        return t3

# Initializing the model
size = 256
m = Model(size=size)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
