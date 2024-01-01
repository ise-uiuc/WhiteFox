
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:31]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = x4[:, 0:31]
        return [v4, v5]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224, 224, 3)
x2 = torch.randn(1, 200, 200, 3)
x3 = torch.randn(1, 200, 200, 3)
x4 = torch.randn(1, 192, 300, 3)
