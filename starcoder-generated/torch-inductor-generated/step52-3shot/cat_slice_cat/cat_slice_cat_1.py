
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        s2 = slice(v1, 0)
        v3 = slice(s2, 0, size)
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 9, 4)
x2 = torch.randn(2, 8, 4)
