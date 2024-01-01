
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *args):
        l = [t for t in args]
        v1 = torch.cat(l, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:0]
        v4 = torch.cat([v1, v3], dim=1)
        return v1, v2, v3, v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
x2 = torch.randn(1, 8, 16, 16)
x3 = torch.randn(1, 8, 8, 8)
x4 = torch.randn(1, 8, 4, 4)
