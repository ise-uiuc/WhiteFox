

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x3, x2):
        v1 = torch.cat([x2, x1], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:x1._shape[0]]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 125, 232)
x2 = torch.randn(1, 254, 232, 232)
x3 = torch.randn(1, 128, 232, 232)
