
class Model(torch.nn.Module):
    def __init__(self, size=3):
        super().__init__()

    def forward(self, x3, x4):
        v3 = torch.cat([x3, x4], dim=1)
        v4 = v3[:, 0:9223372036854775807]
        v5 = v4[:, 0:3]
        v6 = torch.cat([v3, v5], dim=1)
        return v6

# Initializing the model
m = Model(3)

# Inputs to the model
x3 = torch.randn(1, 10, 8, 8)
x4 = torch.randn(1, 7, 8, 8)
