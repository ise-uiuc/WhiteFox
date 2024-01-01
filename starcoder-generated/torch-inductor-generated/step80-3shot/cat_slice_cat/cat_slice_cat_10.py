
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:13]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initialize model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 128, 128)
y1 = torch.randn(1, 20, 64, 64)
z1 = torch.randn(1, 30, 128, 128)
a1 = torch.randn(1, 40, 32, 32)
b1 = torch.randn(1, 50, 512, 512)
