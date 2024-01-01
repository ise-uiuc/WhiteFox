
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
        v1 = [x1, x2, x3, x4, x5]
        v2 = torch.cat(v1, dim=1)
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:size]
        v5 = torch.cat([v2, v4], dim=1)
        return (v5, v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 63, 255)
x2 = torch.randn(1, 1, 55, 255)
x3 = torch.randn(1, 1, 11, 255)
x4 = torch.randn(1, 1, 3, 255)
x5 = torch.randn(1, 1, 25, 255)
