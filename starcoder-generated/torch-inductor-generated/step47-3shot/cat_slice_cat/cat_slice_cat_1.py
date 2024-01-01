
class Model(torch.nn.Module):
    def __init__(self):
        pass
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:8]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

x1 = torch.randn(1, 128)
x2 = torch.randn(1, 64)
x3 = torch.randn(1, 128)
