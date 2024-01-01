
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, :9223372036854775807]
        v3 = v2[:-1]
        v4 = torch.cat([v1, v1], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1000, 64, 64)
x2 = torch.randn(1, 50, 64, 64)
x3 = torch.randn(1, 12, 64, 64)
