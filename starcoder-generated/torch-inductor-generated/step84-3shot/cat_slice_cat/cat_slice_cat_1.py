
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:len(x2)]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model and inputs
m = Model()
x1 = torch.randn(2, 5, 64, 64)
x2 = torch.randn(2, 1, 64, 64)
