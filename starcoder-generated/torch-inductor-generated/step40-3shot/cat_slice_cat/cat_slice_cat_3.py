
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        v1 = torch.cat([x, y, z], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:257]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 256, 96, 96)
y = torch.randn(1, 128, 96, 96)
z = torch.randn(1, 128, 96, 96)
