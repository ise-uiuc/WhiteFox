
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:1]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = torch.cat([x4, x5, x6], dim=1)
        v6 = v5[:, 0:1]
        v7 = torch.cat([v5, v6], dim=1)
        v8 = torch.cat([v4, v7], dim=1)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 148, 53)
x2 = torch.randn(1, 64, 74, 26)
x3 = torch.randn(1, 64, 37, 13)
x4 = torch.randn(1, 64, 64, 32)
x5 = torch.randn(1, 64, 32, 16)
x6 = torch.randn(1, 64, 16, 8)
