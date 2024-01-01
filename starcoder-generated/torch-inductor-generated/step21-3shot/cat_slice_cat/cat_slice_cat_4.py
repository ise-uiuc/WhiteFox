
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x1):
        t1 = torch.cat([x1, x1], dim=1)
 
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:self.size]
 
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model(3)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 2, 3, 64, 64)
