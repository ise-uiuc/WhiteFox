
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:393216]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = torch.cat([x4, x5], dim=1)
        v6 = torch.cat([v4, v5], dim=1)
        return v1, v2, v3, v4, v5, v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 393216)
x2 = torch.randn(1, 8787584)
x3 = torch.randn(1, 393216)
x4 = torch.randn(1, 9223372036854775807)
x5 = torch.randn(1, 9223372036854775807)
