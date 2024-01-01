
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:1024]
        v4 = torch.cat([x4, v3, x5], dim=1)
        v5 = torch.cat([x6, x7, x8, x9], dim=1)
        return v4, v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 1024)
x2 = torch.randn(1, 8, 1024)
x3 = torch.randn(1, 8, 1024)
x4 = torch.randn(1, 16, 1024)
x5 = torch.randn(1, 16, 1024)
x6 = torch.randn(1, 16, 1024)
x7 = torch.randn(1, 16, 1024)
x8 = torch.randn(1, 16, 1024)
x9 = torch.randn(1, 16, 1024)
