
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.cat([x1, x1], dim=1)
        v2 = v1[:, 0:torch.iinfo(torch.int64).max]
        v3 = v2[:, 0:6]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 6, 3)
x2 = torch.randn(8, 6, 3)
