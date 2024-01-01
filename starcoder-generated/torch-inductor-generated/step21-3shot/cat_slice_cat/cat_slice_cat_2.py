
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:-1]
        v3 = v1[:, 0:int(v1.size(1))]
        v4 = torch.cat([v1, v3], dim=1)
        return v4, v5, v6, v7, v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100, 512)
x2 = torch.randn(1, 100, 768)
x3 = torch.randn(1, 100, 256)
x4 = torch.randn(1, 100, 64)
x5 = torch.randn(1, 100, 16)
