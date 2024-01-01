
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:18446744073709551615.0]
        v3 = v2[:, 0:torch.iinfo(torch.int64).max]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 15, 64, 64)
x2 = torch.randn(1, 5, 64, 64)
x3 = torch.randn(1, 8, 64, 64)
