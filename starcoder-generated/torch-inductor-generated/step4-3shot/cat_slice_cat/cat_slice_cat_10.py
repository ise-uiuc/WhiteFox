
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x1, x2):
        c1 = torch.cat([x1, x2], dim=1)
        s1 = c1[:, 0:9223372036854775807]
        s2 = s1[:, 0:self.size]
        c2 = torch.cat([c1, s2], dim=1)
        return c2

# Initializing the model
m = Model(size=0)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
