
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        c1 = torch.cat([x1, x2, x3, x4], dim=1)
        c2 = c1[:, 0:9223372036854775807]
        s1 = c2[:, 0:x2.shape[-1]]
        c3 = torch.cat([c1, s1], dim=1)
        return c3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
x2 = torch.randn(1, 1, 3, 3)
x3 = torch.randn(1, 1, 3, 3)
x4 = torch.randn(1, 1, 3, 3)
x5 = torch.randn(1, 1, 3, 3)
