
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=3)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 50:50+10]
        v4 = torch.cat([v1, v3], dim=3)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 256, 100, 100)
x2 = torch.randn(10, 256, 10, 10)
