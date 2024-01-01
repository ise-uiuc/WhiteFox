
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.cat(x2, x1, dim=1)
        v2 = v1[:, 0: 2147483647]
        v3 = v2[:, 0:2]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 1)
x3 = torch.randn(1, 1)
