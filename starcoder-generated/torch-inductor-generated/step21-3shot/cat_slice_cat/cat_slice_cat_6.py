
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat([x1, x2, x3, x4], dim=1)
        v2 = v1[:, 0:2147483647]
        v3 = v2[:, 0:2147483647]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 7, 32, 32)
x2 = torch.randn(1, 7, 32, 32)
x3 = torch.randn(1, 7, 32, 32)
x4 = torch.randn(1, 7, 32, 32)
