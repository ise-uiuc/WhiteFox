
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:43]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 43, 112, 112)
x2 = torch.randn(10, 43, 110, 110)
x3 = torch.randn(10, 43, 111, 111)
x4 = torch.randn(10, 43, 92, 92)
x5 = torch.randn(10, 43, 91, 91)
