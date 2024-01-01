
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.split(x1, 2, dim=2)
        v2 = v1[0]
        v3 = v1[1]
        v4 = v1[0] * 5.0 + v2
        v5 = torch.cat([v4, v3], dim=2)
        v6 = v5 * 1.6 - 1.9
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 64)
