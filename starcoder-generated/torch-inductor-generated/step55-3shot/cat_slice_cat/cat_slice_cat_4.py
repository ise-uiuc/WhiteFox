
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:sys.maxsize]
        v3 = v2[:, 0:x1.size(3)]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 8, 16)
x2 = torch.randn(1, 5, 8, 16)
x3 = torch.randn(1, 6, 8, 16)
