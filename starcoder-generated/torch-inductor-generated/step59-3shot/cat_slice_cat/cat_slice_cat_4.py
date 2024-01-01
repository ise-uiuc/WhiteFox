
class Model(torch.nn.Module):
    def forward(self, x, y):
        v1 = torch.cat((x, y), dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2, 3, 4)
y = torch.randn(1, 1, 3, 4)
