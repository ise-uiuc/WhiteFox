
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x2, x3):
        v1 = torch.cat([x2, x3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:9223372036854775807]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8223372036854775807, 2, 1)
x3 = torch.randn(1, 2, 3, 4)
