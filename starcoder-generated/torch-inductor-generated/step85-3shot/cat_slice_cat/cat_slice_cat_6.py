
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:-1]
        v3 = torch.cat([v2, x3], dim=1)
        v4 = v3[:, size:]
        v5 = torch.cat([x4, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(2, 3, 64, 64)
x3 = torch.randn(3, 3, 64, 64)
x4 = torch.randn(4, 3, 64, 64)
