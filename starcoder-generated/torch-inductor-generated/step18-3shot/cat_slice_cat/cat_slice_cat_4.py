
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, xa, xb, xc):
        v1 = torch.cat([xa, xb])
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 33554432:67108864]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
xa = torch.randn(1, 30, 15, 15)
xb = torch.randn(1, 23, 13, 13)
xc = torch.randn(1, 17, 11, 11)
