
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x6, x7):
        v1 = torch.cat([x1, x2, x6, x7], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:34]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 34, 64, 64)
x2 = torch.randn(1, 2000, 56, 56)
x6 = torch.randn(1, 256, 28, 28)
x7 = torch.randn(1, 512, 14, 14)
