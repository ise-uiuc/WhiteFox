
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, size):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:1000000000000000000000]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
x2 = torch.randn(1, 15, 64, 64)
size = 1000000000000000000000000
