
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        v1 = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:8]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = torch.cat([x7, x8], dim=1)
        v6 = torch.cat([v4, v5], dim=1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
x2 = torch.randn(1, 10, 64, 64)
x3 = torch.randn(1, 20, 64, 64)
x4 = torch.randn(1, 50, 64, 64)
x5 = torch.randn(1, 100, 64, 64)
x6 = torch.randn(1, 200, 64, 64)
x7 = torch.randn(1, 400, 64, 64)
x8 = torch.randn(1, 512, 64, 64)
