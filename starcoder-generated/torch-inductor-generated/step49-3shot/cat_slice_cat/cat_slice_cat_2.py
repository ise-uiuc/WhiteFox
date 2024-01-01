
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = torch.cat([x3, x4], dim=1)
        v3 = torch.cat([x5, v1], dim=1)
        v4 = v3[:, 0:9223372036854775807]
        v5 = v4[:, 0:9223372036854775807]
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9223372036854775807, 64, 64)
x2 = torch.randn(1, 9223372036854775807, 64, 64)
x3 = torch.randn(1, 9223372036854775807, 64, 64)
x4 = torch.randn(1, 9223372036854775807, 64, 64)
x5 = torch.randn(1, 9223372036854775807, 64, 64)
