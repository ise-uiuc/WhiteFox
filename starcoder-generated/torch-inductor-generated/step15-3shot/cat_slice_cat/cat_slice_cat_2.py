
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:x1.shape[2]]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = torch.cat([v1, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(8, 32767, 32767)
x2 = torch.randn(8, 32767, 32767)
x3 = torch.randn(8, 32767, 32767)
