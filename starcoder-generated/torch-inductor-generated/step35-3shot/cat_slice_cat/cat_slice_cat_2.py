
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, ::2] # The following slice is equivalent to v2 = torch.narrow(v1, dim=1, start=0, length=self.size, end=None)
        v3 = v2[:, 0:self.size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model(7)

# Inputs to the model
x1 = torch.randn(1, 2*3*4, 64, 64)
x2 = torch.randn(1, 9, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
