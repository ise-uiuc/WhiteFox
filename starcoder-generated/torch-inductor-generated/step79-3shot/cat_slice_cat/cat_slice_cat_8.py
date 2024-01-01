
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.cat([x1, x1, x1], dim=1)
        v2 = v1[:, 0:36028797018963970]
        v3 = v2[:, 0:38426849195166273]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 256, 800)
x2 = torch.randn(1, 16, 256, 800)
