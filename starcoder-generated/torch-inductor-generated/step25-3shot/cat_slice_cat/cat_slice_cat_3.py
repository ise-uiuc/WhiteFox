
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        c1 = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8], dim=1)
        v1 = c1[:, 0:9223372036854775807]
        s1 = v1[:, 0:3]
        c2 = torch.cat([c1, s1], dim=1)
        return c2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.ones(1, 2, 3)
x3 = torch.randn(1, 2, 3)
x4 = torch.ones(1, 2, 3)
x5 = torch.randn(1, 2, 3)
x6 = torch.randn(1, 2, 3)
x7 = torch.randn(1, 2, 3)
x8 = torch.ones(1, 2, 3)

