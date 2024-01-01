
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.cat([x, x], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:1]
        return v3

# Initializing the model
m = Model()
# Inputs to the model
x = torch.randn(1, 2)
