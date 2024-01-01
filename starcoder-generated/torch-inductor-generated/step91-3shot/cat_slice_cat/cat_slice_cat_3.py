
class Model(torch.nn.Module):
    def forward(self, x10, x11):
        v10 = torch.cat([x10, x11], dim=1)
        v11 = v10[:, 0:9223372036854775807]
        v12 = v11[:, 0:1728]
        v13 = torch.cat([v10, v12], dim=1)
        return v13

# Initializing the model
m = Model()

# Inputs to the model
x10 = torch.randn(1, 3, 1, 9223372036854775807)
x11 = torch.randn(1, 3, 1, 1728)
