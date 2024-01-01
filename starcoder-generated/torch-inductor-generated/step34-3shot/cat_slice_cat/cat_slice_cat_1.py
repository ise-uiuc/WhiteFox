
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:x1.size(3)]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = torch.cat([x3, x4], dim=1)
        v6 = torch.cat([v5, x5], dim=1)
        v7 = torch.cat([v4, v6], dim=1)
        return v7

# Initializing the model
m1 = Model()

# Inputs to the model
x1 = torch.randn(1, 10000000, 100)
x2 = torch.randn(1, 10000000, 1911)
x3 = torch.randn(1, 10000000, 22)
x4 = torch.randn(1, 10000000, 3112)
x5 = torch.randn(1, 10000000, 4)
