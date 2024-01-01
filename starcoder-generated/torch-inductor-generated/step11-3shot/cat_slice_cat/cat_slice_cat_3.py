
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 32768:32770]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model (Note: x1 and x2 cannot have the same list elements for now)
x1 = torch.randn(32, 64)
x2 = torch.randn(32, 64)
x3 = torch.randn(32, 16, 100)
x4 = torch.randn(32, 16, 256)
