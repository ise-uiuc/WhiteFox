
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = torch.cat([x1, x1], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:13]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 3, 32, 64)
