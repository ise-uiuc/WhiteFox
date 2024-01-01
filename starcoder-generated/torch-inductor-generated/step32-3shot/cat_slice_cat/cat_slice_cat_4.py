
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        c1 = torch.cat([x1, x2, x3, x4], dim=1)
        s1 = c1[:, 0:9223372036854775807]
        s2 = s1[:, 0:50]
        c2 = torch.cat([c1, s2], dim=1)
        return c2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 3, 3, 3)
x2 = torch.randn(1, 2, 3, 3, 3)
x3 = torch.randn(2, 1, 3, 3, 3)
x4 = torch.randn(2, 2, 3, 3, 3)
