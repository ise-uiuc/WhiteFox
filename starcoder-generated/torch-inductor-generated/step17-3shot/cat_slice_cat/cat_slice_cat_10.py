
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        t = torch.cat([x1, x2], dim=1)
        v1 = t[:, 0:9223372036854775807]
        v2 = t[:, 0:9223372036854775807]
        v = torch.cat([t, v1], dim=1)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 14, 14)
x2 = torch.randn(1, 512, 14, 14)
