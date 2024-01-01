
class Model(torch.nn.Module):
    def forward(self, x1, size):
        v1 = torch.cat([x1, x1])
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v1])
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
size = 6000
