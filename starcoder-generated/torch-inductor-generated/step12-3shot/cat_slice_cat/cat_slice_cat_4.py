
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat([x1, x2, x3, x4], dim=1)
        v2 = v1[:, 0:536870911]
        v3 = v1[:, 536870912:]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 256, 256)
x2 = torch.randn(1, 1, 256, 256)
x3 = torch.randn(1, 1, 256, 256)
x4 = torch.randn(1, 1, 256, 256)
x5 = torch.randn(1, 1, 256, 256)
x6 = torch.randn(1, 1, 256, 256)
x7 = torch.randn(1, 1, 256, 256)
x8 = torch.randn(1, 1, 256, 256)
x9 = torch.randn(1, 1, 256, 256)
x10 = torch.randn(1, 1, 256, 256)
x11 = torch.randn(1, 1, 256, 256)
x12 = torch.randn(1, 1, 256, 256)
x13 = torch.randn(1, 1, 256, 256)
x14 = torch.randn(1, 1, 256, 256)
x15 = torch.randn(1, 1, 256, 256)
x16 = torch.randn(1, 1, 256, 256)
x17 = torch.randn(1, 1, 256, 256)
x18 = torch.randn(1, 1, 256, 256)
x19 = torch.randn(1, 1, 256, 256)
x20 = torch.randn(1, 1, 256, 256)
