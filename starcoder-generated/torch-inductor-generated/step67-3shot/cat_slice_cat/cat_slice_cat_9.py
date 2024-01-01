
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *x1):
        v1 = torch.cat(x1, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:12]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 12, 32, 48, 64)
x2 = torch.randn(1, 96, 13, 34, 24)
x3 = torch.randn(1, 64, 25, 12, 13)
x4 = torch.randn(3, 74, 36, 25, 19)
