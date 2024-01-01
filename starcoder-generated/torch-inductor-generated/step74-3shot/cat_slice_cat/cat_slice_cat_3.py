
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, s=64, k=3):
        v = torch.cat([x1, x2, x3, x4], dim=1)
        v = v[:, 0:9223372036854775807]
        v = v[:, 0:s * s * k]
        return torch.cat([v, v], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2048, 4, 4)
x2 = torch.randn(1, 256, 8, 8)
x3 = torch.randn(1, 256, 16, 16)
x4 = torch.randn(1, 256, 32, 32)

# Parameters of the model
s = 4
k = 3

